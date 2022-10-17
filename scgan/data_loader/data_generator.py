import sys
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data as data_utils
import torchvision.transforms as transforms
import pickle as pkl

from scgan.utils.data_utils import *


# Dataset (Input Pipeline)
class CustomDataset(data_utils.Dataset):
    def __init__(self, config, pickle_path, is_training=True):
        self.config = config
        self.is_training = is_training

        with open(pickle_path, 'rb') as f:
            data = pkl.load(f)

        self.word_data = data['word_data']
        self.idx_to_id = {i: w_id for i, w_id in enumerate(self.word_data.keys())}
        self.char_map = data['char_map']

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.word_data)

    def __getitem__(self, idx):
        item = {}
        w_id = self.idx_to_id[idx]

        # Get image and label
        lab, img = self.word_data[w_id]

        img = self.transforms(img / 255.)

        item['img'] = img.float()
        item['label'] = torch.tensor(lab)

        return item


class DataLoader:
    def __init__(self, config, pickle_path):
        self.config = config
        self.pickle_path = pickle_path

    def create_train_loader(self):
        self.dataset = CustomDataset(self.config, self.pickle_path)
        return torch.utils.data.DataLoader(
            self.dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=6, pin_memory=True, collate_fn=self.batch_collate)

    def batch_collate(self, batch):
        items = {}
        max_w = max([item['img'].shape[2] for item in batch])

        # Remove channel dimension, swap height and width, pad widths and return to the original shape
        imgs = [
            F.pad(item['img'], [0, max_w - item['img'].size(2), 0, self.config.img_h - item['img'].size(1)])
            for item in batch
        ]
        items['img'] = torch.stack(imgs)

        items['label_len'] = torch.tensor([len(item['label']) for item in batch])
        items['label'] = pad_sequence([item['label'] for item in batch], batch_first=True, padding_value=0)

        return items

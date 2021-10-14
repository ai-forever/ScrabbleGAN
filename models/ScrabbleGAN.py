import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from models.model_utils import BigGAN as BGAN
from utils.data_utils import *
import pandas as pd


def recognizer_block1(in_channels, out_channels, kernel_size, pad):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad),
        nn.ReLU(True),
        nn.MaxPool2d(2)
    )


def recognizer_block2(in_channels, out_channels, kernel_size, pad):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


def recognizer_block3(in_channels, out_channels, kernel_size, pad):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=pad),
        nn.ReLU(True),
        nn.MaxPool2d((2, 2), (2, 1), (0, 1))
    )


def R_arch(channels):
    arch = {}
    arch[32] = nn.Sequential(
        recognizer_block1(channels, 64, 3, 1),
        recognizer_block1(64, 128, 3, 1),
        recognizer_block2(128, 256, 3, 1),
        recognizer_block3(256, 256, 3, 1),
        recognizer_block2(256, 512, 3, 1),
        recognizer_block3(512, 512, 3, 1),
        recognizer_block2(512, 512, 2, 0)
    )
    arch[64] = nn.Sequential(
        recognizer_block1(channels, 64, 3, 1),
        recognizer_block1(64, 128, 3, 1),
        recognizer_block2(128, 256, 3, 1),
        recognizer_block3(256, 256, 3, 1),
        recognizer_block2(256, 512, 3, 1),
        recognizer_block3(512, 512, 3, 1),
        recognizer_block2(512, 512, 2, 0),
        recognizer_block3(512, 512, 2, 0),
    )
    arch[128] = nn.Sequential(
        recognizer_block1(channels, 64, 3, 1),
        recognizer_block1(64, 128, 3, 1),
        recognizer_block2(128, 256, 3, 1),
        recognizer_block3(256, 256, 3, 1),
        recognizer_block2(256, 512, 3, 1),
        recognizer_block3(512, 512, 3, 1),
        recognizer_block2(512, 512, 2, 0),
        recognizer_block3(512, 512, 2, 0),
        recognizer_block3(512, 512, 2, 0),
    )
    return arch


class Recognizer(nn.Module):
    def __init__(self, cfg, num_chars):
        super(Recognizer, self).__init__()
        self.convs = R_arch(cfg.channels)[cfg.img_h]
        self.output = nn.Linear(512, num_chars)
        self.prob = nn.LogSoftmax(dim=2)

    def forward(self, x):
        out = self.convs(x)

        out = out.squeeze(2)  # [b, c, w]
        out = out.permute(0, 2, 1)  # [b, w, c]

        # Predict for len(num_chars) classes at each timestep
        out = self.output(out)
        out = self.prob(out)
        return out


class ScrabbleGAN(nn.Module):
    def __init__(self, cfg, char_map, lexicon_paths):
        super().__init__()

        self.z_dist = torch.distributions.Normal(loc=0, scale=1.)
        self.z_dim = cfg.z_dim

        self._load_lexicon(cfg, char_map, lexicon_paths)

        self.batch_size = cfg.batch_size
        self.num_chars = len(char_map)
        self.word_map = WordMap(char_map)

        self.batch_size = cfg.batch_size
        self.config = cfg

        self.R = Recognizer(cfg, self.num_chars)
        self.G = BGAN.Generator(resolution=cfg.img_h, G_shared=cfg.g_shared,
                                bn_linear=cfg.bn_linear, n_classes=self.num_chars, hier=True, channels=cfg.channels)
        self.D = BGAN.Discriminator(resolution=cfg.img_h, bn_linear=cfg.bn_linear, n_classes=self.num_chars, channels=cfg.channels)

    def _load_lexicon(self, cfg, char_map, lexicon_paths):
        """Get word list from lexicon to be used to generate fake images."""
        fake_words = []
        for lexicon_path in lexicon_paths:
            with open(lexicon_path, 'r') as f:
                fake_words.extend(f.read().splitlines())

        fake_words_clean = []
        for word in fake_words:
            word_set = set(word)
            if len(word_set.intersection(char_map.keys())) == len(word_set):
                fake_words_clean.append(word)
        self.fake_words = fake_words_clean
        self.fake_y_dist = torch.distributions.Categorical(
            torch.tensor([1. / len(self.fake_words)] * len(self.fake_words)))

    def forward_fake(self, z=None, fake_y=None, b_size=None):
        b_size = self.batch_size if b_size is None else b_size

        # If z is not provided, sample it
        if z is None:
            self.z = self.z_dist.sample([b_size, self.z_dim]).to(self.config.device)
        else:
            self.z = z.repeat(b_size, 1).to(self.config.device)

        # If fake words are not provided, sample it
        if fake_y is None:
            # Sample lexicon indices, get words, and encode them using char_map
            sample_lex_idx = self.fake_y_dist.sample([b_size])
            fake_y = [self.fake_words[i] for i in sample_lex_idx]

        self.fake_y_decoded = fake_y
        fake_y, fake_y_lens = self.word_map.encode(fake_y)
        self.fake_y_lens = fake_y_lens.to(self.config.device)

        # Convert y into one-hot
        self.fake_y = fake_y.to(self.config.device)
        self.fake_y_one_hot = F.one_hot(fake_y, self.num_chars).to(self.config.device)

        self.fake_img = self.G(self.z, self.fake_y_one_hot)


def create_model(config, char_map, lexicon_path):
    model = ScrabbleGAN(config, char_map, lexicon_path)
    model.to(config.device)

    return model

import torch
import argparse
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from importlib import import_module
import shutil
import cv2
import os
import numpy as np
import logging

import torch.nn.functional as F
from data_loader.data_generator import DataLoader
from utils.data_utils import *
from utils.training_utils import ModelCheckpoint
from losses_and_metrics import loss_functions, metrics
from config import Config

seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
np.random.seed(seed)

level = logging.INFO
format_log = '%(message)s'

os.makedirs('output', exist_ok=True)
handlers = [logging.FileHandler('output/output.log'), logging.StreamHandler()]
logging.basicConfig(level=level, format=format_log, handlers=handlers)


def get_train_loader(config, data_pkl_path):
    data_loader = DataLoader(config, data_pkl_path)
    data_loader = data_loader.create_train_loader()
    return data_loader


class Trainer:
    def __init__(self, config, args):
        self.config = config
        self.terminal_width = shutil.get_terminal_size((80, 20)).columns
        logging.info(f' Loading Data '.center(self.terminal_width, '*'))

        self.is_unlabeled_data = False
        if args.unlabeled_pkl_path:
            self.is_unlabeled_data = True
            logging.info("Using unlabeled data to train discriminator")

        self.labeled_loader = get_train_loader(config, args.data_pkl_path)
        self.num_chars = len(self.labeled_loader.dataset.char_map)
        if self.is_unlabeled_data:
            self.unlabeled_loader = get_train_loader(config, args.unlabeled_pkl_path)

        # Model
        logging.info(f' Model: {self.config.architecture} '.center(self.terminal_width, '*'))
        model_type = import_module('models.' + self.config.architecture)
        create_model = getattr(model_type, 'create_model')
        self.model = create_model(self.config, self.labeled_loader.dataset.char_map, args.lexicon_path)
        logging.info(f'{self.model}\n')
        self.model.to(self.config.device)

        self.word_map = WordMap(self.labeled_loader.dataset.char_map)

        # Loss, Optimizer and LRScheduler
        self.G_criterion = getattr(loss_functions, self.config.g_loss_fn)('G')
        self.D_criterion = getattr(loss_functions, self.config.d_loss_fn)('D')
        self.R_criterion = getattr(loss_functions, self.config.r_loss_fn)()
        self.G_optimizer = torch.optim.Adam(self.model.G.parameters(), lr=self.config.g_lr, betas=self.config.g_betas)
        self.D_optimizer = torch.optim.Adam(self.model.D.parameters(), lr=self.config.d_lr, betas=self.config.d_betas)
        self.R_optimizer = torch.optim.Adam(self.model.R.parameters(), lr=self.config.r_lr, betas=self.config.r_betas)
        self.optimizers = [self.G_optimizer, self.D_optimizer, self.R_optimizer]

        # Use a linear learning rate decay but start the decay only after specified number of epochs
        lr_decay_lambda = lambda epoch: (1. - (1. / self.config.epochs_lr_decay)) \
            if epoch > (epoch - self.config.epochs_lr_decay - 1) else 1.
        self.schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lr_decay_lambda) for opt in self.optimizers]

        # Metric
        # self.metric = getattr(metrics, config.metric)()

        self.start_epoch = 1
        # Load checkpoint if training is to be resumed
        self.model_checkpoint = ModelCheckpoint(config=self.config)
        if args.pretrain_path:
            self.model, self.optimizers, self.schedulers, self.start_epoch = \
                self.model_checkpoint.load(
                    self.model, args.pretrain_path, self.optimizers, self.schedulers, load_only_R=args.load_only_R)
            self.G_optimizer, self.D_optimizer, self.R_optimizer = self.optimizers
            logging.info(f'Resuming model training from epoch {self.start_epoch}')

        # logging
        self.writer = SummaryWriter(f'logs')

    def set_requires_grad(self, nets, requires_grad=False):
        """
        Source - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/fd29199c33bd95704690aaa16f238a4f8e74762c/models/base_model.py
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_G(self):
        """Completes forward, backward, and optimize for G"""
        # generate fake image using generator
        self.model.forward_fake()
        # Switch off backpropagation for R and D
        self.set_requires_grad([self.model.D, self.model.R], False)

        # Generator loss will be determined by the evaluation of generated image by discriminator and recognizer
        pred_D_fake = self.model.D(self.model.fake_img)
        pred_R_fake = self.model.R(self.model.fake_img).permute(1, 0, 2)  # [w, b, num_chars]

        self.loss_G = self.G_criterion(pred_D_fake)
        self.loss_R_fake = self.R_criterion(pred_R_fake, self.model.fake_y,
                                            torch.ones(pred_R_fake.size(1)).int() * pred_R_fake.size(0),
                                            self.model.fake_y_lens)
        self.loss_R_fake = torch.mean(self.loss_R_fake[~torch.isnan(self.loss_R_fake)])

        # the below part has been mostly copied from - https://github.com/amzn/convolutional-handwriting-gan/blob/2cfbc794cca299445e5ba070c8634b6cd1a84261/models/ScrabbleGAN_baseModel.py#L345
        self.loss_G_total = self.loss_G + self.config.grad_alpha * self.loss_R_fake
        grad_fake_R = torch.autograd.grad(self.loss_R_fake, self.model.fake_img, retain_graph=True)[0]
        self.loss_grad_fake_R = 10 ** 6 * torch.mean(grad_fake_R ** 2)
        grad_fake_adv = torch.autograd.grad(self.loss_G, self.model.fake_img, retain_graph=True)[0]
        self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
        if self.config.grad_balance:
            epsilon = 10e-50
            self.loss_G_total.backward(retain_graph=True)
            grad_fake_R = torch.autograd.grad(self.loss_R_fake, self.model.fake_img,
                                              create_graph=True, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.model.fake_img,
                                                create_graph=True, retain_graph=True)[0]
            a = self.config.grad_alpha * torch.div(torch.std(grad_fake_adv), epsilon + torch.std(grad_fake_R))
            self.loss_R_fake = a.detach() * self.loss_R_fake
            self.loss_G_total = self.loss_G + self.loss_R_fake
            self.loss_G_total.backward(retain_graph=True)
            grad_fake_R = torch.autograd.grad(self.loss_R_fake, self.model.fake_img,
                                              create_graph=False, retain_graph=True)[0]
            grad_fake_adv = torch.autograd.grad(self.loss_G, self.model.fake_img,
                                                create_graph=False, retain_graph=True)[0]
            self.loss_grad_fake_R = 10 ** 6 * torch.mean(grad_fake_R ** 2)
            self.loss_grad_fake_adv = 10 ** 6 * torch.mean(grad_fake_adv ** 2)
            with torch.no_grad():
                self.loss_G_total.backward()
        else:
            self.loss_G_total.backward()

        self.G_optimizer.step()
        self.G_optimizer.zero_grad()

    def optimize_D_unlabeled(self):
        """Completes forward, backward, and optimize for D on unlabeled data"""
        # generate fake image using generator
        self.model.forward_fake()
        # Switch on backpropagation for R and D
        self.set_requires_grad([self.model.D], True)
        pred_D_fake = self.model.D(self.model.fake_img.detach())
        pred_D_real = self.model.D(self.real_unlabeled_img.detach())
        # we will now calculate discriminator loss for both real and fake images
        self.loss_D_fake = self.D_criterion(pred_D_fake, 'fake')
        self.loss_D_real = self.D_criterion(pred_D_real, 'real')
        self.loss_D = self.loss_D_fake + self.loss_D_real

        self.loss_D.backward()
        self.D_optimizer.step()
        self.D_optimizer.zero_grad()

    def optimize_D_R(self):
        """Completes forward, backward, and optimize for D and R"""
        # generate fake image using generator
        self.model.forward_fake()
        # Switch on backpropagation for R and D
        self.set_requires_grad([self.model.D, self.model.R], True)

        pred_D_fake = self.model.D(self.model.fake_img.detach())
        pred_D_real = self.model.D(self.real_img.detach())

        # we will now calculate discriminator loss for both real and fake images
        self.loss_D_fake = self.D_criterion(pred_D_fake, 'fake')
        self.loss_D_real = self.D_criterion(pred_D_real, 'real')
        self.loss_D = self.loss_D_fake + self.loss_D_real

        # recognizer
        self.pred_R_real = self.model.R(self.real_img).permute(1, 0, 2)  # [w, b, num_chars]
        self.loss_R_real = self.R_criterion(self.pred_R_real, self.real_y,
                                            torch.ones(self.pred_R_real.size(1)).int() * self.pred_R_real.size(0),
                                            self.real_y_lens)
        self.loss_R_real = torch.mean(self.loss_R_real[~torch.isnan(self.loss_R_real)])

        self.loss_D_and_R = self.loss_D + self.loss_R_real

        self.loss_D_and_R.backward()

        self.D_optimizer.step()
        self.R_optimizer.step()
        self.D_optimizer.zero_grad()
        self.R_optimizer.zero_grad()

    def train(self):
        logging.info(f' Training '.center(self.terminal_width, '*'))

        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            logging.info(f' Epoch [{epoch}/{self.config.num_epochs}] '.center(self.terminal_width, 'x'))
            self.model.train()

            train_loader = self.labeled_loader
            len_loader = len(self.labeled_loader)
            if self.is_unlabeled_data:
                if len(self.unlabeled_loader) > len(self.labeled_loader):
                    train_loader = zip(cycle(self.labeled_loader), self.unlabeled_loader)
                else:
                    train_loader = zip(self.labeled_loader, cycle(self.unlabeled_loader))

            progbar = tqdm(train_loader, total=len_loader)
            losses_G, losses_D, losses_D_real, losses_D_fake = [], [], [], []
            losses_R_real, losses_R_fake, grads_fake_R, grads_fake_adv = [], [], [], []
            for i, batch_items in enumerate(progbar):
                if self.is_unlabeled_data:
                    batch_items, data_unlabeled = batch_items
                    self.real_unlabeled_img = data_unlabeled['img'].to(self.config.device)

                self.real_img = batch_items['img'].to(self.config.device)
                self.real_y = batch_items['label'].to(self.config.device)
                self.real_y_one_hot = F.one_hot(batch_items['label'], self.num_chars).to(self.config.device)
                self.real_y_lens = batch_items['label_len'].to(self.config.device)

                # Forward + Backward + Optimize G
                if (i % self.config.train_gen_steps) == 0:
                    # optimize generator
                    self.optimize_G()

                # Forward + Backward + Optimize D and R
                self.optimize_D_R()

                if self.is_unlabeled_data:
                    self.optimize_D_unlabeled()

                # save losses
                losses_G.append(self.loss_G.cpu().data.numpy())
                losses_D.append(self.loss_D.cpu().data.numpy())
                losses_D_real.append(self.loss_D_real.cpu().data.numpy())
                losses_D_fake.append(self.loss_D_fake.cpu().data.numpy())
                losses_R_real.append(self.loss_R_real.cpu().data.numpy())
                losses_R_fake.append(self.loss_R_fake.cpu().data.numpy())
                grads_fake_R.append(self.loss_grad_fake_R.cpu().data.numpy())
                grads_fake_adv.append(self.loss_grad_fake_adv.cpu().data.numpy())

                progbar.set_description("G = %0.3f, D = %0.3f, R_real = %0.3f, R_fake = %0.3f,  " %
                                        (np.mean(losses_G), np.mean(losses_D),
                                         np.mean(losses_R_real), np.mean(losses_R_fake)))

            logging.info(f'G = {np.mean(losses_G):.3f}, D = {np.mean(losses_D):.3f}, '
                         f'R_real = {np.mean(losses_R_real):.3f}, R_fake = {np.mean(losses_R_fake):.3f}'
            )
            # Save one generated fake image from last batch
            img = self.model.fake_img.cpu().data.numpy()[0]
            normalized_img = ((img + 1) * 255 / 2).astype(np.uint8)
            normalized_img = np.moveaxis(normalized_img, 0, -1)
            cv2.imwrite(f'./output/epoch_{epoch}_fake_img.png', normalized_img)

            # Print Recognizer prediction for 4 (or batch size) real images from last batch
            num_imgs = 4 if self.config.batch_size >= 4 else self.config.batch_size
            labels = self.word_map.decode(self.real_y[:num_imgs].cpu().numpy())
            preds = self.word_map.recognizer_decode(self.pred_R_real.max(2)[1].permute(1, 0)[:num_imgs].cpu().numpy())
            logging.info('\nRecognizer predictions for real images:')
            max_len_label = max([len(i) for i in labels])
            for lab, pred in zip(labels, preds):
                logging.info(f'Actual: {lab:<{max_len_label+2}}|  Predicted: {pred}')

            # Print Recognizer prediction for 4 (or batch size) fake images from last batch
            logging.info('Recognizer predictions for fake images:')
            labels = self.word_map.decode(self.model.fake_y[:num_imgs].cpu().numpy())
            preds_R_fake = self.model.R(self.model.fake_img).permute(1, 0, 2).max(2)[1].permute(1, 0)
            preds = self.word_map.recognizer_decode(preds_R_fake[:num_imgs].cpu().numpy())
            max_len_label = max([len(i) for i in labels])
            for lab, pred in zip(labels, preds):
                logging.info(f'Actual: {lab:<{max_len_label+2}}|  Predicted: {pred}')

            # Change learning rate according to scheduler
            for sch in self.schedulers:
                sch.step()

            # save checkpoint after every 5 epochs
            if epoch % 5 == 0:
                self.model_checkpoint.save(self.model, epoch, self.G_optimizer, self.D_optimizer, self.R_optimizer,
                                           *self.schedulers)

            # write logs
            self.writer.add_scalar(f'loss_G', np.mean(losses_G), epoch * i)
            self.writer.add_scalar(f'loss_D/fake', np.mean(losses_D_fake), epoch * i)
            self.writer.add_scalar(f'loss_D/real', np.mean(losses_D_real), epoch * i)
            self.writer.add_scalar(f'loss_R/fake', np.mean(losses_R_fake), epoch * i)
            self.writer.add_scalar(f'loss_R/real', np.mean(losses_R_real), epoch * i)
            self.writer.add_scalar(f'grads/fake_R', np.mean(grads_fake_R), epoch * i)
            self.writer.add_scalar(f'grads/fake_adv', np.mean(grads_fake_adv), epoch * i)

        self.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_pkl_path", required=True, type=str,
                        help="Path to the pickle processed data")
    parser.add_argument("--unlabeled_pkl_path", type=str, default='',
                        help="Path to the pickle processed unlabeled data")
    parser.add_argument("--pretrain_path", type=str, default='',
                        help="Path to the pretrain model weights")
    parser.add_argument("--load_only_R", action='store_true',
                        help="To load only Recognizer from model weights")
    parser.add_argument("--lexicon_path", action='append', required=True,
                        type=str, help="Path to the lexicon txt. Can be passed "
                        "multiple times")
    args = parser.parse_args()

    trainer = Trainer(Config, args)
    trainer.train()

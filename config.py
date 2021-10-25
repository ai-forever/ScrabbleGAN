import torch


class Config:
    # arch[img_h] defines the architecture to be selected
    # imh_h and char_width should be in: 32x16, 64x32, 128x64
    img_h = 128
    char_w = 64
    channels = 3

    batch_size = 8
    num_epochs = 1000
    epochs_lr_decay = 100  # learning rate decay will be applied for last these many steps (should be <= num_epochs)

    train_gen_steps = 4  # generator weights to be updated after every specified number of steps
    grad_alpha = 1
    grad_balance = True

    architecture = 'ScrabbleGAN'

    # Generator and Discriminator networks
    bn_linear = 'SN'
    g_shared = False

    g_lr = 2e-4
    d_lr = 2e-4
    r_lr = 2e-4
    g_betas = [0., 0.999]
    d_betas = [0., 0.999]
    r_betas = [0., 0.999]
    g_loss_fn = 'HingeLoss'
    d_loss_fn = 'HingeLoss'
    r_loss_fn = 'CTCLoss'

    # Noise vector
    z_dim = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

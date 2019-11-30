import torch as t
from torch import nn


class Regress(nn.Module):
    def __init__(self, probabilistic=False, in_chan=3, batch_norm=False):
        super(Regress, self).__init__()
        n_out = 1 + int(probabilistic)
        self.conv_blocks = nn.Sequential(
            ResMod(in_chan=in_chan, out_chan=in_chan, batch_norm=batch_norm),
            ResMod(in_chan=in_chan, out_chan=in_chan, batch_norm=batch_norm),
            nn.Conv2d(in_chan, 16, 3, stride=2),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.Conv2d(64, 128, 3, stride=2),
        )
        if probabilistic:
            s = 210
        else:
            s = 190
        adjustment = s // 2 ** 4

        # Output layers
        self.dense_unit = nn.Sequential(nn.Linear(128 * adjustment ** 2, n_out * 16),
                                        nn.Linear(n_out * 16, n_out * 8),
                                        nn.Linear(n_out * 8, n_out * 4),
                                        nn.Linear(n_out * 4, n_out * 2),
                                        nn.Linear(n_out * 2, n_out))

        self.frozen = False

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        y = self.dense_unit(out)
        return y


class ResMod(nn.Module):
    def __init__(self, batch_norm=False, in_chan=1, out_chan=1):
        super(ResMod, self).__init__()

        self.in_chan = in_chan
        self.out_chan = out_chan

        ###############
        # first block #
        ###############
        module = nn.ModuleList()

        module.append(nn.Conv2d(in_chan, out_chan, (3, 3), padding=1))

        if batch_norm:
            module.append(nn.BatchNorm2d(out_chan))

        module.append(nn.ReLU())

        self.first_block = module

        ################
        # second block #
        ################
        module = nn.ModuleList()

        module.append(nn.Conv2d(out_chan, out_chan, (3, 3), padding=1))

        if batch_norm:
            module.append(nn.BatchNorm2d(out_chan))

        self.second_block = module

        self.shortcut = nn.Conv2d(in_chan, out_chan, (1, 1))

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        for module in self.first_block:
            x = module(x)
        for module in self.second_block:
            x = module(x)

        x += self.shortcut(residual)

        x = self.relu(x)

        return x

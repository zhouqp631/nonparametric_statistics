import argparse
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
device = torch.device("cpu")

batch_size = 64
mean = torch.load('.\statistics\mnist_mean.pt')
(full_dim, mid_dim, hidden) = (1 * 28 * 28, 1000, 5)
transform = torchvision.transforms.ToTensor()
trainset = torchvision.datasets.MNIST(root='.\data\MNIST',train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)


def dequantize(x):
    '''Dequantize data.

    Add noise sampled from Uniform(0, 1) to each pixel (in [0, 255]).

    Args:
        x: input tensor.
        reverse: True in inference mode, False in training mode.
    Returns:
        dequantized data.
    '''
    noise = torch.distributions.Uniform(0., 1.).sample(x.size())
    return (x * 255. + noise) / 256.

def prepare_data(x, mean=None, reverse=False):
    """Prepares data for NICE.

    In training mode, flatten and dequantize the input.
    In inference mode, reshape tensor into image size.

    Args:
        x: input minibatch.
        zca: ZCA whitening transformation matrix.
        mean: center of original dataset.
        reverse: True if in inference mode, False if in training mode.
    Returns:
        transformed data.
    """
    if reverse:
        assert len(list(x.size())) == 2
        [B, W] = list(x.size())
        assert W == 1 * 28 * 28
        x += mean
        x = x.reshape((B, 1, 28, 28))
    else:
        assert len(list(x.size())) == 4
        [B, C, H, W] = list(x.size())
        assert [C, H, W] == [1, 28, 28]
        x = dequantize(x)
        x = x.reshape((B, C*H*W))
        x -= mean
    return x
#%%
# import matplotlib.pyplot as plt
# n, bins, patches = plt.hist(x[:10].flatten().numpy(), 50, density=True, facecolor='g', alpha=0.75)
# plt.show()
# y = dequantize(x)
# y = y.reshape(-1,784)-mean
# n, bins, patches = plt.hist(y[:10].flatten().numpy(), 50, density=True, facecolor='g', alpha=0.75)
# plt.show()
#%%
class StandardLogistic(torch.distributions.Distribution):
    def __init__(self):
        super(StandardLogistic, self).__init__()

    def log_prob(self, x):
        """Computes data log-likelihood.

        Args:
            x: input tensor.
        Returns:
            log-likelihood.
        """
        return -(F.softplus(x) + F.softplus(-x))

    def sample(self, size):
        """Samples from the distribution.

        Args:
            size: number of samples to generate.
        Returns:
            samples.
        """
        z = torch.distributions.Uniform(0., 1.).sample(size)
        return torch.log(z) - torch.log(1. - z)



"""Additive coupling layer.
"""
class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config
        self.in_block = nn.Sequential(nn.Linear(in_out_dim//2, mid_dim),nn.ReLU())
        self.mid_block = nn.ModuleList([nn.Sequential(nn.Linear(mid_dim, mid_dim),nn.ReLU())for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim//2)
    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W//2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)

        if reverse:
            on = on - shift
        else:
            on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W))

"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(torch.zeros((1, dim)),
                                  requires_grad=True)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det_J = torch.sum(self.scale)
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x, log_det_J

"""NICE main model.
"""
class NICE(nn.Module):
    def __init__(self, prior, coupling,in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a NICE.

        Args:
            prior: prior distribution over latent space Z.
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(NICE, self).__init__()
        self.prior = prior
        self.in_out_dim = in_out_dim
        self.coupling = nn.ModuleList([
            Coupling(in_out_dim=in_out_dim,mid_dim=mid_dim,hidden=hidden,
                     mask_config=(mask_config+i)%2) \
                     for i in range(coupling)])
        self.scaling = Scaling(in_out_dim)

    def g(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, reverse=True)
        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        for i in range(len(self.coupling)):
            x = self.coupling[i](x)
        return self.scaling(x)

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.in_out_dim))
        return self.g(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)



prior = StandardLogistic()
coupling = 4
mask_config = 1
flow = NICE(prior=prior,
            coupling=coupling,
            in_out_dim=full_dim,
            mid_dim=mid_dim,
            hidden=hidden,
            mask_config=mask_config).to(device)
optimizer = torch.optim.Adam(flow.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-4)

total_iter = 0
train = True
running_loss = 0

max_iter = 50
mean_loss_every_100_steps = []
for iter in range(max_iter):
    steps = 0
    for _, data in tqdm(enumerate(trainloader, 1)):
        inputs, _ = data
        inputs = prepare_data(inputs, mean=mean).to(device)

        # log-likelihood of input minibatch
        loss = -flow(inputs).mean()
        running_loss += float(loss)

        # backprop and update parameters
        optimizer.zero_grad()  # clear gradient tensor
        loss.backward()
        optimizer.step()
        steps += 1
        if steps % 100 == 0:
            mean_loss = running_loss / 100
            print('iter %s:' % iter,'loss = %.3f' % mean_loss)
            mean_loss_every_100_steps.append(mean_loss)
            running_loss = 0.0



            with torch.no_grad():
                z, _ = flow.f(inputs)
                reconst = flow.g(z).cpu()
                reconst = prepare_data(reconst, mean=mean, reverse=True)
                samples = flow.sample(20).cpu()
                samples = prepare_data(samples, mean=mean, reverse=True)
                torchvision.utils.save_image(torchvision.utils.make_grid(reconst),'./reconstruction/'  +'iter%d_step%d.png' % (iter,steps),nrow=4)
                torchvision.utils.save_image(torchvision.utils.make_grid(samples),'./samples/'  +'iter%d_step%d.png' % (iter,steps),nrow=4)


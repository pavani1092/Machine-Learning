#!/usr/bin/env python

# Generative Adversarial Networks (GAN) example in PyTorch.
# See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt


# Data params
raw_data = loadmat('fft.mat')
raw_data = raw_data['y']
# fft_data = np.array((raw_data.real, raw_data.imag))


# Model params
g_input_size = 2    # Random noise dimension coming into generator, per output vector
g_hidden_size = 50   # Generator complexity
g_output_size = 2    # size of generated output vector

d_input_size = 2   # Minibatch size - cardinality of distributions
d_hidden_size = 50   # Discriminator complexity
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
# minibatch_size = d_input_size

d_learning_rate = 2e-4  # 2e-4
g_learning_rate = 2e-4
optim_betas = (0.9, 0.999)
num_epochs = 30000
print_interval = 200
d_steps = 1  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator
g_steps = 1

(name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)

print("Using data [%s]" % name)


# def distribution_sample():
#     result = np.dstack((fft_data[0], fft_data[1])).flatten()
#     result2 = np.reshape(result, (len(raw_data), 4))
#     return torch.Tensor(result2)


def noise():
    return torch.Tensor(np.random.rand(len(raw_data), 2))  # Uniform-dist data into generator, _NOT_ Gaussian


# ##### MODELS: Generator model and discriminator model

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = torch.sigmoid(self.map2(x))
        return self.map3(x)


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.map1(x))
        x = F.elu(self.map2(x))
        return torch.sigmoid(self.map3(x))


G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size)
D = Discriminator(input_size=d_input_func(d_input_size), hidden_size=d_hidden_size, output_size=d_output_size)
criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss
d_optimizer = optim.Adam(D.parameters(), lr=d_learning_rate, betas=optim_betas)
g_optimizer = optim.Adam(G.parameters(), lr=g_learning_rate, betas=optim_betas)

# dataset = distribution_sample()
dataset = torch.Tensor(raw_data)
g_fake_data = None

# For Plots
t = np.arange(0, num_epochs, 1,dtype=int)
G_error = np.zeros(num_epochs)

for epoch in range(num_epochs):
    for d_index in range(d_steps):
        # 1. Train D on real+fake
        D.zero_grad()

        #  1A: Train D on real
        d_real_data = Variable(dataset)
        d_real_decision = D(d_real_data)
        d_real_error = criterion(d_real_decision, Variable(torch.ones(len(d_real_decision), 1)))  # ones = true
        d_real_error.backward()  # compute/store gradients, but don't change params

        #  1B: Train D on fake
        d_gen_input = Variable(noise())
        d_fake_data = G(d_gen_input).detach()  # detach to avoid training G on these labels
        d_fake_decision = D(preprocess(d_fake_data))
        d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(len(d_fake_decision), 1)))  # zeros = fake
        d_fake_error.backward()
        d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

    for g_index in range(g_steps):
        # 2. Train G on D's response (but DO NOT train D on these labels)
        G.zero_grad()

        gen_input = Variable(noise())
        g_fake_data = G(gen_input)
        dg_fake_decision = D(preprocess(g_fake_data))
        g_error = criterion(dg_fake_decision, Variable(torch.ones(len(raw_data), 1)))  # we want to fool, so pretend it's all genuine
        G_error[epoch] = g_error.item()

        g_error.backward()
        g_optimizer.step()  # Only optimizes G's parameters
        print(epoch, " : D: ", d_real_error.item(), " G: ", G_error[epoch])

fake_data = g_fake_data.detach().numpy()
savemat('ifft.mat', mdict={'iF': fake_data})
plt.ioff()
fig = plt.figure()
plt.plot(t, G_error)
plt.xlabel('Epoch')
plt.ylabel('Generator Error')
plt.savefig('test0.png')
plt.close(fig)



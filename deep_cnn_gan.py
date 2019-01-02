#!/usr/bin/python3
# https://arxiv.org/pdf/1511.06434.pdf
# https://medium.com/@keisukeumezawa/dcgan-generate-the-images-with-deep-convolutinal-gan-55edf947c34b

# no maxpooling, just convs with stride = 2 - for 2x2 px from input, next layer got 1x1 px
# batch normalization - mean = 0, variance = 1

import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def visualize(loader):
  it = iter(loader)
  images, labels = it.next()

  fig = plt.figure(figsize=(25, 4))
  plot_size = 20
  for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.transpose(images[idx], (1,2,0)))
    ax.set_title(str(labels[idx].item()))

  plt.show()


def view_samples(epoch, samples):
  fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
  for ax, img in zip(axes.flatten(), samples[epoch]):
    img = img.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    im = ax.imshow(img.reshape((32,32,3)))
  plt.show()

def scale(x, feature_range=(-1,1)):
  # x = x * torch.Tensor([np.sum(np.abs(feature_range))]) + torch.Tensor([feature_range[0]])
  min, max = feature_range
  x = x * (max - min) + min
  return x


def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
  layers = []
  conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

  layers.append(conv_layer)
  if batch_norm:
    layers.append(nn.BatchNorm2d(out_channels))
  
  return nn.Sequential(*layers)


def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
  layers = []
  deconv_layer = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

  layers.append(deconv_layer)
  if batch_norm:
    layers.append(nn.BatchNorm2d(out_channels))

  return nn.Sequential(*layers)


def real_loss(D_out, smooth=False):
  batch_size = D_out.size(0)
  if smooth:
    labels = torch.ones(batch_size) * 0.95
  else:
    labels = torch.ones(batch_size)
  
  labels = labels.cuda()
  criterion = nn.BCEWithLogitsLoss()
  loss = criterion(D_out.squeeze(), labels)
  return loss


def fake_loss(D_out):
  batch_size = D_out.size(0)
  labels = torch.zeros(batch_size).cuda()
  criterion = nn.BCEWithLogitsLoss()  
  loss = criterion(D_out.squeeze(), labels)
  return loss


class Discriminator(nn.Module):
  def __init__(self, conv_dim=32):
    super(Discriminator, self).__init__()

    self.conv_dim = conv_dim

    self.conv1 = conv(3, conv_dim * 4, kernel_size=4, stride=2, padding=1, batch_norm=False)
    self.conv2 = conv(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=2, padding=1, batch_norm=True)
    self.conv3 = conv(conv_dim * 8, conv_dim * 16, kernel_size=4, stride=2, padding=1, batch_norm=True)    
    
    self.fc1 = nn.Linear(conv_dim * 16 * 4 * 4, 1)

  def forward(self, x):
    
    x = F.leaky_relu(self.conv1(x), 0.2)
    x = F.leaky_relu(self.conv2(x), 0.2)
    x = F.leaky_relu(self.conv3(x), 0.2)

    x = x.view(-1, self.conv_dim * 16 * 4 * 4)

    x = self.fc1(x)

    return x


class Generator(nn.Module):
  def __init__(self, z_size, conv_dim=32):
    super(Generator, self).__init__()

    self.z_size = z_size
    self.conv_dim = conv_dim

    self.fc1 = nn.Linear(z_size, conv_dim * 16 * 4 * 4)

    self.deconv1 = deconv(conv_dim * 16, conv_dim * 8)
    self.deconv2 = deconv(conv_dim * 8, conv_dim * 4)
    self.deconv3 = deconv(conv_dim * 4, 3, batch_norm=False)

  def forward(self, x):
    
    x = self.fc1(x)

    x = x.view(-1, conv_dim*16, 4, 4)

    x = F.relu(self.deconv1(x))
    x = F.relu(self.deconv2(x))
    x = F.tanh(self.deconv3(x))
    
    return x


transform = transforms.ToTensor()

svhn_train = datasets.SVHN(root='data/', split='train', download=True, transform=transform)

batch_size = 128
num_workers = 0

train_loader = torch.utils.data.DataLoader(dataset=svhn_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# visualize(train_loader)

conv_dim = 16
z_size = 100

D = Discriminator(conv_dim)
G = Generator(z_size=z_size, conv_dim=conv_dim)

train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
  G.cuda()
  D.cuda()

lr = 0.0002
beta1 = 0.5
beta2 = 0.999

d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])

epochs = 100

samples = []
losses = []

print_every = 300

sample_size = 16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).cuda().float()

# for epoch in range(epochs):
#   for batch_i, (real_images, _) in enumerate(train_loader):

#     batch_size = real_images.size(0)

#     real_images = scale(real_images)
#     real_images = real_images.cuda()

#     d_optimizer.zero_grad()

#     D_real = D(real_images)
#     d_real_loss = real_loss(D_real)

#     z = np.random.uniform(-1, 1, size=(batch_size, z_size))
#     z = torch.from_numpy(z).cuda().float()
#     fake_imgs = G(z)

#     D_fake = D(fake_imgs)
#     d_fake_loss = fake_loss(D_fake)

#     d_loss = d_fake_loss + d_real_loss
#     d_loss.backward()
#     d_optimizer.step()

#     z = np.random.uniform(-1, 1, size=(batch_size, z_size))
#     z = torch.from_numpy(z).cuda().float()
#     fake_imgs = G(z)

#     D_fake = D(fake_imgs)
#     g_loss = real_loss(D_fake)

#     g_loss.backward()
#     g_optimizer.step()

#     if batch_i % print_every == 0:
#       losses.append((d_loss.item(), g_loss.item()))
#       print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
#         epoch+1, epochs, d_loss.item(), g_loss.item()))

#   G.eval()
#   fixed_z = fixed_z.cuda()
#   samples_z = G(fixed_z)
#   samples.append(samples_z)
#   G.train()

# with open('train_samples.pkl', 'wb') as f:
#   pkl.dump(samples, f)

with open('train_samples.pkl', 'rb') as f:
  samples = pkl.load(f)

# fig, ax = plt.subplots()
# losses = np.array(losses)
# plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
# plt.plot(losses.T[1], label='Generator', alpha=0.5)
# plt.title("Training Losses")
# plt.legend()

view_samples(-1, samples)
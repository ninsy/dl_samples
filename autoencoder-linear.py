# https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f

import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def imshow(x):
  x = x.numpy()
  plt.imshow(x.transpose((1,2,0)))
  plt.show()


class Autoencoder(nn.Module):
  def __init__(self, encoding_dim):
    super(Autoencoder, self).__init__()

    self.fc1 = nn.Linear(784, encoding_dim)
    self.fc2 = nn.Linear(encoding_dim, 784)

  def forward(self, x):
    
    x = x.view(-1, 784)
    x = F.relu(self.fc1(x))
    x = F.sigmoid(self.fc2(x))

    return x

def train(model, train_loader):
  epochs = 15

  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  model.train()
  for epoch in range(1, epochs+1):

    train_loss = 0.0

    for images, _ in train_loader:
      
      images = images.to('cuda')
      images = images.view(batch_size, -1)      

      optimizer.zero_grad()
      output = model(images)

      loss = criterion(output, images)
      loss.backward()
      optimizer.step()

      train_loss += loss.item() * images.size(0)

    train_loss = train_loss/len(train_loader)
    print('Epoch: {}\tTraining loss: {:.6f}'.format(epoch, train_loss))
    
  return model
    

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

batch_size = 20

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

it = iter(test_loader)
images, _ = it.next()
# imshow(torchvision.utils.make_grid(images))

encoding_dim = 32
model = Autoencoder(encoding_dim)
model.to('cuda')

model = train(model, train_loader)

flattened_images = images.view(images.size(0), -1)
flattened_images = flattened_images.to('cuda')

output = model(flattened_images)
output = output.view(batch_size, 1, 28, 28)
output = output.detach().cpu().numpy()

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

for images, row in zip([images, output], axes):
  for img, ax in zip(images, row):
    print(img.shape)
    ax.imshow(np.squeeze(img), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import numpy as np

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
  
    # nn.Conv2d(input_channels, output_channels, kernel_size)
    
    # 32x32x3 image tensor
    self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
    # 16x16x16 image tensor    
    self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
    # 8x8x32 image tensor        
    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    
    # tensor x and y got halved in every conv2d because of maxPool2d
    # after third conv layer - 4x4x64 image tensor
    
    self.pool = nn.MaxPool2d(2,2)

    self.fc1 = nn.Linear(64 * 4 * 4, 500)
    self.fc2 = nn.Linear(500, 10)

    self.dropout = nn.Dropout(0.20)


  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))    
    x = self.pool(F.relu(self.conv3(x)))    
        
    # no longer conv, no longer using matrices, 
    # starting to use fully connected layers - need to flatten into vector
    x = x.view(-1, 64 * 4 * 4)

    x = self.dropout(x)
    x = F.relu(self.fc1(x))
    
    x = self.dropout(x)
    x = F.relu(self.fc2(x))        
    return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_workers = 0
batch_size = 16
valid_size = 0.2
epochs = 50
valid_loss_min = np.Inf

transform = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(10),
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)

num_train = len(train_data)
indicies = list(range(num_train))
np.random.shuffle(indicies)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indicies[split:], indicies[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model = Net()
model.to(device)

model.load_state_dict(torch.load('model_cifar.pt'))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
  train_loss = 0.0
  valid_loss = 0.0

  model.train()
  for data, target in train_loader:
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model.forward(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    train_loss += loss.item() * data.size(0)

  model.eval()  
  for data, target in valid_loader:
    data, target = data.to(device), target.to(device) 
    output = model.forward(data)
    loss = criterion(output, target)
    valid_loss += loss.item() * data.size(0)

  train_loss = train_loss / len(train_loader.dataset)
  valid_loss = valid_loss / len(valid_loader.dataset)  

  print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
    epoch, train_loss, valid_loss))

  # save model if validation loss has decreased
  if valid_loss <= valid_loss_min:
    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
      valid_loss_min,
      valid_loss))
    torch.save(model.state_dict(), 'model_cifar.pt')
    valid_loss_min = valid_loss


test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.load_state_dict(torch.load('model_cifar.pt'))
model.eval()
for data, target in test_loader:
  data, target = data.to(device), target.to(device) 
  output = model.forward(data)
  loss = criterion(output, target)
  test_loss += loss.item() * data.size(0)

  _, pred = torch.max(output, 1)
  correct_tensor = pred.eq(target.data.view_as(pred))
  correct = np.squeeze(correct_tensor.cpu().numpy())
  for i in range(batch_size):
    label = target.data[i]
    class_correct[label] += correct[i].item()
    class_total[label] += 1


test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
  if class_total[i] > 0:
      print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
          classes[i], 100 * class_correct[i] / class_total[i],
          np.sum(class_correct[i]), np.sum(class_total[i])))
  else:
      print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
  100. * np.sum(class_correct) / np.sum(class_total),
  np.sum(class_correct), np.sum(class_total)))
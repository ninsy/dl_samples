#  paper link https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
# http://genekogan.com/works/style-transfer/

import torch
import torch.optim as optim
from torchvision import models, transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

def gram_matrix(tensor):
  """Gram matrix of a given tensor"""
  # batch_size, depth, height, width
  _, d , h, w = tensor.size()
  tensor = tensor.view(d, h * w)
  gram = torch.mm(tensor, tensor.t())
  return gram

def get_features(image, model, layers=None):
  """ Run image forward through a model to get features for set of layers
      Default layers are for VGGNet matching Gatys et al (2016)    
  """

  if layers is None:
    layers = {
      '0': 'conv1_1',
      '5': 'conv2_1',
      '10': 'conv3_1',
      '19': 'conv4_1',
      '21': 'conv4_2',      
      '28': 'conv5_1',
    }

  features = {}
  x = image
  for name, layer in model._modules.items():
    x = layer(x)
    if name in layers:
      features[layers[name]] = x

  return features


def im_convert(tensor):
  image = tensor.to('cpu').clone().detach()
  image = image.numpy().squeeze()  
  image = image.transpose(1,2,0)
  image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
  image = image.clip(0, 1)
  return image
      

def load_image(img_path, max_size=400, shape=None):
  image =  Image.open(img_path).convert('RGB')

  if max(image.size) > max_size:
    size = max_size
  else:
    size = max(image.size)

  if shape is not None:
    size = shape

  in_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize(
      (0.485, 0.456, 0.406),
      (0.229, 0.224, 0.225)
    )
  ])

  # discard alpha channel ( :3 ) and add batch dimension
  image = in_transform(image)[:3,:,:].unsqueeze(0)
  return image

# no need for classifier
vgg = models.vgg19(pretrained=True).features

# only optimizing target image
for param in vgg.parameters():
  param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

content = load_image('images/misia.jpg').to(device)
style = load_image('images/sunflowers.jpg', shape=content.shape[-2:]).to(device)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# ax1.imshow(im_convert(content))
# ax2.imshow(im_convert(style))
# fig.show()

content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

style_grams = {
  layer: gram_matrix(style_features[layer]) for layer in style_features
}

target = content.clone().requires_grad_(True).to(device)

style_weights = {
  'conv1_1': 1.0,
  'conv2_1': 0.9,
  'conv3_1': 0.5,
  'conv4_1': 0.4,
  'conv5_1': 0.4,  
}

content_weight = 1
style_weight = 1000

show_every = 400

optimizer = optim.Adam([target], lr=0.003)
steps = 4000

for ii in range(1, steps+1):
  
  target_features = get_features(target, vgg)
  content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
  style_loss = 0

  for layer in style_weights:
    target_feature = target_features[layer]
    target_gram = gram_matrix(target_feature)
    
    _, d, h, w = target_feature.shape

    style_gram = style_grams[layer]

    layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2) 

    style_loss += layer_style_loss / (d * h * w)

  total_loss = style_weight * style_loss + content_weight * content_loss

  optimizer.zero_grad()
  total_loss.backward()
  optimizer.step()

  if ii % show_every == 0:
    date_str = time.strftime("%Y-%m-%d-%H:%M")
    print('{} Total loss: {}'.format(date_str, total_loss.item()))
    # plt.imshow(im_convert(target))
    # plt.show()

date_str = time.strftime("%Y-%m-%d-%H:%M")
plt.imsave('target{}.jpg'.format(date_str), im_convert(target))

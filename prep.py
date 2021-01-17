from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import math

def load_img(path, max_size=400):
    img = Image.open(path).convert('RGB')

    size = min(max(img.size), max_size)

    input_trans = transforms.Compose([
        transforms.Resize((size, int(1.5*size))),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.456), (0.229, 0.224, 0.225))
    ])

    img_trans = input_trans(img)[:3, :, :].unsqueeze(0)

    return img_trans


def tensor_to_img(tensor):
    img = tensor.to("cpu").clone().detach()
    img = img.numpy().squeeze()
    img = img.transpose(1, 2, 0)
    img = img * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406))
    img = img.clip(0, 1)
    out_img = Image.fromarray((img*255).astype(np.uint8))
    return out_img


def get_features(img, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1'}
    features = {}
    x = img
    for name, layer in enumerate(model.features):
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features


def gram_matrix(x):
    batch_size, channels, hgt, wdth = x.size()
    tensor = x.view(channels, hgt*wdth)
    gram = torch.mm(tensor, tensor.t())
    return gram


def generate_mask(shape, num_points: int): # будем создавать маску рандомно на основе диаграммы Вороного
    if num_points > 30: # ограничим количество точек для генерации диаграммы Вороного
        num_points = 30

    points_x = np.random.randint(0, shape[0], num_points)
    points_y = np.random.randint(0, shape[1], num_points)
    points = np.stack((points_x, points_y))
    points_bool = np.random.randint(0, 2, num_points)

    mask = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            min_dist = math.sqrt(shape[0]**2+shape[1]**2)
            for n in range(num_points):
                point = points[:,n]
                dist = math.sqrt(((i - point[0])**2 + (j - point[1])**2))
                if dist < min_dist:
                    min_dist = dist
                    min_index = n
            mask[i,j] = points_bool[min_index]

    return torch.tensor(mask)


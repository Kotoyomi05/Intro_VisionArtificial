import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. Definir las transformaciones para las imágenes (ResNet requiere 224x224 y normalización de ImageNet)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Aumentación de datos básica
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'dataset'

# 2. Cargar los datasets usando ImageFolder
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

# 3. Crear los DataLoaders para alimentar el modelo por lotes
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print("¡Clases detectadas:", class_names)
print("Tamaños del dataset:", dataset_sizes)
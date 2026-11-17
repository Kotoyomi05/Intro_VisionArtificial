import torch
import torchvision.models as models
import torch.nn as nn

# 1. Cargar el modelo ResNet18 con los pesos preentrenados de ImageNet
# (Usamos 'weights' en lugar de 'pretrained' que está deprecado en versiones recientes)
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# 2. Congelar las capas convolucionales base (Freezing)
# Esto evita que se desentrenen los filtros que ya reconocen bordes, formas y texturas
for param in model.parameters():
    param.requires_grad = False

# 3. Reemplazar la última capa (classifier head) para tus propias clases
# 'model.fc.in_features' toma las 512 características de salida y las conecta con tus clases (ej: 3 clases)
num_classes = 4
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Nota: Solo los parámetros de 'model.fc' tendrán requires_grad = True por defecto, 
# lo que permite entrenar únicamente la cabeza (Fine-Tuning).

print("¡Modelo base cargado, congelado y adaptado con éxito!")
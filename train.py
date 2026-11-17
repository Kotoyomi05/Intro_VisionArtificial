import torch
import torch.nn as nn
import torch.optim as optim
from modelo_trasnfer import model # Importa tu modelo desde el archivo que ya hiciste
from dataset_loader import dataloaders, dataset_sizes # Importa los datos

# 1. Configurar el dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en dispositivo: {device}")

model = model.to(device)

# 2. Definir la función de pérdida y el optimizador
# CrossEntropyLoss es ideal para clasificación multiclase
criterion = nn.CrossEntropyLoss()

# Optimizador: Solo entrenamos los parámetros de la capa final (model.fc) gracias al freezing
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 3. Ciclo de entrenamiento (Training Loop básico)
num_epochs = 5  # Puedes empezar con 5 épocas para probar rápido

for epoch in range(num_epochs):
    print(f'Época {epoch+1}/{num_epochs}')
    print('-' * 10)

    # Fase de entrenamiento
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Poner a cero los gradientes
        optimizer.zero_grad()

        # Forward: calcular predicciones del modelo
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward: calcular gradientes y actualizar solo la última capa
        loss.backward()
        optimizer.step()

        # Estadísticas
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['train']
    epoch_acc = float(running_corrects.double() / dataset_sizes['train']) * 100

    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

print("¡Entrenamiento finalizado con éxito!")

# 4. Guardar los pesos del modelo entrenado
torch.save(model.state_dict(), 'modelo_cultivos_rescatado.pth')
print("Modelo guardado como 'modelo_cultivos_rescatado.pth'")
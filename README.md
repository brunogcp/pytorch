<div align="center">
  <h3 align="center">PyTorch</h3>
  <div>
  <a href="https://bgcp.vercel.app/article/ec09e895-b423-41dc-809f-fcfa606a008c">
  <img src="https://img.shields.io/badge/Download PDF (ENGLISH)-black?style=for-the-badge&logoColor=white&color=000000" alt="three.js" />
  </a>
  </div>
</div>

## 🚀 Introdução ao PyTorch em Python

PyTorch é uma biblioteca de aprendizado de máquina para Python, conhecida por sua flexibilidade e dinâmica na construção de modelos de aprendizado profundo. Com uma interface intuitiva e fácil de usar, o PyTorch facilita tanto a pesquisa acadêmica quanto o desenvolvimento de aplicações práticas.

### 🌟 Principais Características:

- **🔥 Computação Dinâmica**: Suporta gráficos computacionais dinâmicos que facilitam alterações em tempo real.
- **📊 Fácil Integração com Dados**: Excelente suporte para carregamento de dados e transformações.
- **💡 Autograd Automático**: Simplifica a diferenciação automática.
- **🚀 Otimizado para GPU**: Maximiza o desempenho utilizando CUDA para cálculos em GPU.

## 🛠️ Instalação

Prepare seu ambiente Python para começar a trabalhar com o PyTorch.

### Windows, Linux e macOS:

1. **Instale o PyTorch**:

   Visite o [site oficial do PyTorch](https://pytorch.org/get-started/locally/) e escolha a configuração adequada para seu ambiente para obter o comando de instalação específico. Um exemplo comum poderia ser:

```bash
pip install torch torchvision torchaudio
```

## 📊 Uso Básico

### Configuração Inicial:

Após a instalação, importe o PyTorch no seu script Python para começar.

```python
import torch
```

### Criando Tensores:

Tensores são a espinha dorsal do PyTorch, usados para armazenar os dados dos modelos.

```python
# Criando um tensor 2x3
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor)
```

### Operações com Tensores:

O PyTorch oferece uma vasta gama de operações para manipulação de tensores.

```python
# Adição
result = tensor + tensor
print(result)

# Multiplicação por escalar
result = tensor * 3
print(result)
```

### Usando GPU:

Se disponível, você pode acelerar as operações utilizando GPU.

```python
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
```

## 📈 PyTorch para Redes Neurais

### Construindo uma Rede Neural Simples:

Vamos construir uma rede neural simples para classificação de dígitos.

1. **Define a Rede Neural**:

   Crie uma classe que herda de `torch.nn.Module` e defina suas camadas na inicialização.

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
print(model)
```

2. **Treinamento da Rede**:

   Prepare seus dados, defina uma função de perda e um otimizador para treinar sua rede.

```python
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Preparando os dados
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Definindo o loop de treinamento
def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(-1, 28*28))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

# Treinando a rede
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(1, 11):  # passa pelo dataset 10 vezes
    train(model, device, train_loader, criterion, optimizer, epoch)

```

### 🔍 Testes:

1. **Validação do Modelo**:
   
   Após o treinamento, teste o desempenho do modelo em um conjunto de dados de validação.

2. **Ajuste de Hiperparâmetros**:
   
   Experimente diferentes taxas de aprendizado, funções de perda e arquiteturas de rede para melhorar o desempenho.

## 🏆 Conclusão

Neste tutorial, você deu seus primeiros passos com o PyTorch, uma ferramenta poderosa para o desenvolvimento de modelos de aprendizado profundo. Desde a criação de tensores até a construção e treinamento de uma rede neural simples, você viu como o PyTorch pode ser intuitivo e eficaz.

Espero que este guia tenha sido divertido e informativo, e que você esteja agora mais preparado para explorar projetos mais complexos com o PyTorch. Continue experimentando e, acima de tudo, divirta-se com a aprendizagem de máquina! 🐍🔥
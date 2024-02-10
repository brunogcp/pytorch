import torch

# Criando um tensor 2x3
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    
print(tensor)


# Adição
result = tensor + tensor
print(result)

# Multiplicação por escalar
result = tensor * 3
print(result)


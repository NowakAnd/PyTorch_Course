"""
01:19:20 czas
https://www.learnpytorch.io/
https://github.com/mrdbourke/pytorch-deep-learning/discussions
"""

import torch

# Introducion to tensors

# Scalar #
scalar = torch.tensor(7)
print("Scalar representation:")
print(scalar)
print(f"Number of dimensions: {scalar.ndim}")
print(f"Tensor as int: {scalar.item()} \n")  # Returning tensor back as Python int

# Vector #
vector = torch.tensor([7, 7])
print("Vector representation:")
print(vector)
print(f"Number of dimensions: {vector.ndim}")
print(f"Vector shape: {vector.shape} \n")

# MATRIX #
MATRIX = torch.tensor([[7, 8],
                       [7, 10]])
print("MATRIX representation:")
print(MATRIX)
print(f"Number of dimensions: {MATRIX.ndim}")
print(MATRIX[0])
print(MATRIX[1])
print(MATRIX.shape)

# TENSOR #
TENSOR = torch.tensor([[[1, 2, 3],
                        [3, 4, 5],
                        [6, 7, 8]]])
print("\nTENSOR representation:")
print(TENSOR)
print(f"Number of dimensions: {TENSOR.ndim}")
print(f"Vector shape: {TENSOR.shape} \n")

'''Random tensors'''
# Create a random tensor of size (3, 4)
random_tensor = torch.rand(3, 4)
print(random_tensor)

# Create a random tensor with similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(3, 224, 224)) # height, width, colour channel
print(random_image_size_tensor.shape, random_image_size_tensor.ndim)

'''Zeros and ones'''
# Create a tensor of all zeros
zeros = torch.zeros(size=(3, 4))
print(zeros)
print(zeros*random_tensor)

#Create a tensor of all ones
ones = torch.ones(size=(3, 4))
print(ones)

'''Creating a range of tensors and tensors-like'''
# Use torch.range()
one_to_ten = torch.arange(1, 11)
print(one_to_ten)

# Creating tensors like
ten_zeros = torch.zeros_like(input=one_to_ten)
print(ten_zeros)

'''Tensor datatypes'''
# Float 32 tensor
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None,  # what datatype is the tensor (eg. float32, float16)
                               device=None,  # what device is your tensor on
                               requires_grad=False)  # whether to track gradients with these tensors operations
print(float_32_tensor.device)
float_16_tensor = float_32_tensor.type(torch.float16)
print(float_16_tensor)

# 2:03:32

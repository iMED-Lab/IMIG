import torch

# ImageNet normalization constants
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def preprocess(x, mean, std):
    """Apply channel-wise normalization."""
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y


def preprocess_input_function(x):
    """Apply ImageNet normalization to input tensor."""
    return preprocess(x, mean=mean, std=std)


def undo_preprocess(x, mean, std):
    """Reverse channel-wise normalization."""
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y


def undo_preprocess_input_function(x):
    """Reverse ImageNet normalization."""
    return undo_preprocess(x, mean=mean, std=std)

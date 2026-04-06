import os
import numpy as np


def make_one_hot(target, target_one_hot):
    """Convert class indices to one-hot encoding in-place."""
    target = target.view(-1, 1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)


def makedir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def print_and_write(str, file):
    """Print to console and write to file."""
    print(str)
    file.write(str + '\n')


def find_high_activation_crop(activation_map, percentile=95):
    """Find the bounding box of the high-activation region in an activation map."""
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1

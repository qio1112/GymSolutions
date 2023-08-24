import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size, device):
    """
    Args:
        input_size: int, the dimension of inputs to be given to the network
        output_size: int, the dimension of the output
        n_layers: int, the number of hidden layers of the network
        size: int, the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.

    TODO:
    Build a feed-forward network (multi-layer perceptron, or mlp) that maps
    input_size-dimensional vectors to output_size-dimensional vectors.
    It should have 'n_layers' layers, each of 'size' units and followed
    by a ReLU nonlinearity. Additionally, the final layer should be linear (no ReLU).

    That is, the network architecture should be the following:
    [LINEAR LAYER]_1 -> [RELU] -> [LINEAR LAYER]_2 -> ... -> [LINEAR LAYER]_n -> [RELU] -> [LINEAR LAYER]

    "nn.Linear" and "nn.Sequential" may be helpful.
    """
    #######################################################
    #########   YOUR CODE HERE - 7-15 lines.   ############
    layers = []
    in_layer = nn.Linear(input_size, size)
    relu = nn.ReLU()
    layers.append(in_layer)
    layers.append(relu)
    for i in range(n_layers):
        layer_i = nn.Linear(size, size)
        layers.append(layer_i)
        layers.append(relu)
    out_layer = nn.Linear(size, output_size)
    layers.append(out_layer)

    return nn.Sequential(*layers).to(device)

    #######################################################
    #########          END YOUR CODE.          ############


def np2torch(x, cast_double_to_float=True, device="cpu"):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x

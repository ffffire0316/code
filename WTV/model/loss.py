import torch
import torch.nn as nn

def lossfunciton(output,target):
    cr = nn.CrossEntropyLoss()
    return cr(output,target)


#!/usr/bin/python3

from torch.autograd import Variable
import torch.nn as nn
import torch 
import sys

class Net(nn.Module):
    pass


# create net and load weights from file 
trained_model = Net()
trained_model.load_state_dict(torch.load(sys.argv[1]))


# Export the trained model to ONNX
#at first we need fake data to evaulate network
dummy_input = Variable(torch.randn(1, 3, 300, 300)) # one black and white 28 x 28 picture will be the input to the model
#now we can export ONNX
torch.onnx.export(trained_model, dummy_input, sys.argv[2])

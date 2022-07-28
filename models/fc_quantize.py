import math
import sys
from .quant import quantization
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple

class fully_connected(nn.Linear):
    def __init__(self,args,in_channels,out_channels,fp=False,bits=32):
        super(fully_connected,self).__init__(in_channels,out_channels)
        self.full_precision = fp
        self.bits = bits
        if self.bits == 32:
            self.full_precision = True
        if not self.full_precision:
            self.quant_actv = quantization(args, 'fm',[1,in_channels,1,1],bits=self.bits)
            self.quant_wght = quantization(args, 'wt', [out_channels,in_channels,1,1],bits=self.bits)
            self.quant_otpt = quantization(args,'ot',[1,out_channels,1,1],bits=self.bits)
            print(f"fc bits: {self.bits}")
    def forward(self, inputs):
        if not self.full_precision:
            weight = self.quant_wght(self.weight)
            inputs = self.quant_actv(inputs)
        else:
            weight = self.weight
        output = F.linear(inputs,weight)
        if not self.full_precision:
            output = self.quant_otpt(output)
        return output


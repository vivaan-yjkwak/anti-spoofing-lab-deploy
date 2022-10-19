import torchvision.models as tmodels
from models.baseresnetwgrl import baseresnet18wgrl
import torch.nn as nn

def getbaseresnet18():
  resnet18 = tmodels.resnet18(pretrained=True)
  resnet18.fc = nn.Linear(512, 2)

  return resnet18

def getbaseresnet18wgrl(numclass, numdclass):
  resnet18 = baseresnet18wgrl(numclass, numdclass)
  return resnet18
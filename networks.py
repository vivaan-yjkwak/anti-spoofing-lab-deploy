from models.myresnet import myresnet18
from models.baseresnet import baseresnet18

def getresnet18():
  resnet18 = myresnet18(pretrained=False, num_classes=2)
  return resnet18

def getbaseresnet18():
  resnet18 = baseresnet18(pretrained=False)
  return resnet18

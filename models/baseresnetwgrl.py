import torch
import torch.nn as nn
# from torch.autograd import Function
import torchvision.models as models
#
# class GradientReversal(Function):
#   @staticmethod
#   def forward(ctx, x, alpha):
#     ctx.save_for_backward(x, alpha)
#     return x
#
#   @staticmethod
#   def backward(ctx, grad_output):
#     grad_input = None
#     _, alpha = ctx.saved_tensors
#     if ctx.needs_input_grad[0]:
#       grad_input = - alpha * grad_output
#     return grad_input, None
#
# revgrad = GradientReversal.apply
#
# class GRL(nn.Module):
#   def __init__(self, alpha):
#     super().__init__()
#     self.alpha = torch.tensor(alpha, requires_grad=False)
#
#   def forward(self, x):
#     return revgrad(x, self.alpha)

class Feature_Generator_ResNet18(nn.Module):
  def __init__(self):
    super(Feature_Generator_ResNet18, self).__init__()
    model_resnet = models.resnet18(pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3

  def forward(self, input):
    feature = self.conv1(input)
    feature = self.bn1(feature)
    feature = self.relu(feature)
    feature = self.maxpool(feature)
    feature = self.layer1(feature)
    feature = self.layer2(feature)
    feature = self.layer3(feature)
    return feature


class Feature_Embedder_ResNet18(nn.Module):
  def __init__(self):
    super(Feature_Embedder_ResNet18, self).__init__()
    model_resnet = models.resnet18(pretrained=True)
    self.layer4 = model_resnet.layer4
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
    self.bottleneck_layer_fc = nn.Linear(512, 512)
    self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
    self.bottleneck_layer_fc.bias.data.fill_(0.1)
    self.bottleneck_layer = nn.Sequential(
      self.bottleneck_layer_fc,
      nn.ReLU(),
      nn.Dropout(0.5)
    )

  def forward(self, input):
    feature = self.layer4(input)
    feature = self.avgpool(feature)
    feature = feature.view(feature.size(0), -1)
    feature = self.bottleneck_layer(feature)
    feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
    feature = torch.div(feature, feature_norm)
    return feature


class Classifier(nn.Module):
  def __init__(self, numclasses):
    super(Classifier, self).__init__()
    self.classifier_layer = nn.Linear(512, numclasses)
    self.classifier_layer.weight.data.normal_(0, 0.01)
    self.classifier_layer.bias.data.fill_(0.0)

  def l2_norm(self, input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

  def forward(self, input):
    self.classifier_layer.weight.data = self.l2_norm(self.classifier_layer.weight, axis=0)
    classifier_out = self.classifier_layer(input)
    return classifier_out

#
# class Discriminator(nn.Module):
#   def __init__(self, numdclasses = 3):
#     super(Discriminator, self).__init__()
#     self.fc1 = nn.Linear(512, 512)
#     self.fc1.weight.data.normal_(0, 0.01)
#     self.fc1.bias.data.fill_(0.0)
#     self.fc2 = nn.Linear(512, numdclasses)
#     self.fc2.weight.data.normal_(0, 0.3)
#     self.fc2.bias.data.fill_(0.0)
#     self.ad_net = nn.Sequential(
#       self.fc1,
#       nn.ReLU(),
#       nn.Dropout(0.5),
#       self.fc2
#     )
#     self.grl_layer = GRL(alpha=1.0)
#
#   def forward(self, feature):
#     adversarial_out = self.ad_net(self.grl_layer(feature))
#     return adversarial_out


class DG_model(nn.Module):
  def __init__(self, numclasses, numdclsses):
    super(DG_model, self).__init__()

    self.backbone = Feature_Generator_ResNet18()
    self.embedder = Feature_Embedder_ResNet18()

    self.classifier = Classifier(numclasses)
    # self.dis = Discriminator(numdclsses)

  def forward(self, input):
    feature = self.backbone(input)
    feature = self.embedder(feature)
    classifier_out = self.classifier(feature)
    # dis_invariant = self.dis(feature)
    return classifier_out#, dis_invariant

def baseresnet18wgrl(numclasses, numdclsses):
  model = DG_model(numclasses, numdclsses)
  return model

if __name__ == '__main__':
  x1 = torch.ones(2, 3, 256, 256)
  model = DG_model(2, 3)
  cls, dis = model(x1)
  print (cls.shape, dis.shape)

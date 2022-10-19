from PIL import Image, ImageOps
import torch
import torch.nn as nn
from networks import getbaseresnet18, getbaseresnet18wgrl
import os
from torchvision.transforms.functional import center_crop, to_tensor
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


"""
Color based anti-spoofing
"""
class KABANGFASEngine:
  def __init__(self):
    self.resize_p1 = (260, 260)
    self.resize_p2 = (244, 324)
    self.centorcrop_p1 = (256, 256)
    self.centorcrop_p2 = (320, 240)
    self.model_p1 = getbaseresnet18()
    self.model_p2 = getbaseresnet18()
    self.pdleK = 11
    self.model_pdle_p1 = getbaseresnet18wgrl(self.pdleK, 7)
    self.probsm = nn.Softmax(dim=1)
    self.regrsteps = torch.linspace(0, 1.0, steps=self.pdleK)

  def loadckpt(self, strckpt_p1, strckpt_p2, strckpt_pdle_p1):
    checkpoint = torch.load(strckpt_p1)
    self.model_p1.load_state_dict(checkpoint['model_state_dict'], strict=True)
    self.model_p1.eval()

    checkpoint = torch.load(strckpt_p2)
    self.model_p2.load_state_dict(checkpoint['model_state_dict'], strict=True)
    self.model_p2.eval()

    checkpoint = torch.load(strckpt_pdle_p1)
    self.model_pdle_p1.load_state_dict(checkpoint['model_state_dict'], strict=False)
    self.model_pdle_p1.eval()

  def deploy(self, t_input_p1, t_input_p2):
    logit = self.model_p1(t_input_p1)
    prob = self.probsm(logit)
    print("patch1 score spoof {:.7f} vs real {:.7f}".format(float(prob[0][0]), float(prob[0][1])))

    logit = self.model_p2(t_input_p2)
    prob = self.probsm(logit)
    print("patch2 score spoof {:.7f} vs real {:.7f}".format(float(prob[0][0]), float(prob[0][1])))

    logit = self.model_pdle_p1(t_input_p1)
    prob = self.probsm(logit)
    expectprob = torch.sum(self.regrsteps * prob, dim=1)
    expectprob = expectprob.detach().numpy()
    print("patch1 pdle expected score spoof {:.7f} vs real {:.7f}".format(1.0-float(expectprob), float(expectprob)))


  def genpatch2tensor(self, pilimg, fdbox):
    ecode1, x_1by1, y_1by1, w_1by1, h_1by1 = self.genXbyYcorrdinate(fdbox, pilimg.width, pilimg.height, "1by1")
    ecode2, x_4by3, y_4by3, w_4by3, h_4by3 = self.genXbyYcorrdinate(fdbox, pilimg.width, pilimg.height, "4by3")

    if ecode1 + ecode2 < 2:
      # print ("gen patch error code 1by1:{} 4by3:{}".format(ecode1, ecode2))
      # error
      return -1

    try:
      pilimg_1by1 = pilimg.crop([x_1by1, y_1by1, x_1by1 + w_1by1, y_1by1 + h_1by1])
      pilimg_cropresize = pilimg_1by1.resize(self.resize_p1)
      cropresize_p1 = center_crop(pilimg_cropresize, self.centorcrop_p1)
      pilimg_4by3 = pilimg.crop([x_4by3, y_4by3, x_4by3 + w_4by3, y_4by3 + h_4by3])
      pilimg_cropresize = pilimg_4by3.resize(self.resize_p2)
      cropresize_p2 = center_crop(pilimg_cropresize, self.centorcrop_p2)
    except:
      # error
      return -1
    # debug
    # cropresize_p1.show()
    # cropresize_p2.show()
    return 1, to_tensor(cropresize_p1), to_tensor(cropresize_p2)

  def genXbyYcorrdinate(self, fdbox, imgw, imgh, XbyY):
    x, y, w, h = fdbox[0], fdbox[1], fdbox[2], fdbox[3]
    x2, y2 = (x + w), (y + h)
    cx = (x + x2) // 2
    cy = (y + y2) // 2

    if XbyY == "1by1":
      halfmaxw = max(w, h) // 2
      halfmaxh = max(w, h) // 2
    elif XbyY == "4by3":
      halfmaxw = max(w, h) // 2
      halfmaxh = max(w, h) // 1.5

    newx = cx - halfmaxw
    newy = cy - halfmaxh
    neww = int(halfmaxw * 2)
    newh = int(halfmaxh * 2)

    if newx < 0 or newy < 0 or (newx + neww) > imgw or (newy + newh) > imgh:
      return -1, 0, 0, 0, 0

    newx = cx - halfmaxw
    newy = cy - halfmaxh
    neww = int(halfmaxw * 2)
    newh = int(halfmaxh * 2)
    return 1, newx, newy, neww, newh

def loadimgandfd(strtestimg):
  strtestwfd = "{}.fd".format(strtestimg)
  pilimg = Image.open(strtestimg)
  pilimg = ImageOps.exif_transpose(pilimg)

  with open(strtestwfd, "r") as the_file:
    strline = the_file.readline()
    the_file.close()
  strtokens = strline.split()
  if len(strtokens) != 4:
    print ("fd format is not collect, do fd again {}".format(strtestimg))

  fdbox = [int(strtokens[0]), int(strtokens[1]), int(strtokens[2]), int(strtokens[3])]
  return pilimg, fdbox


def deploycolorfas(strtestimg):
  # strckpt_p1 = "/home/user/model_2022/v220922/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db_221010_ifUcCDxJv533hBgeYAvV3C_bsize256_optadam_lr0.0001_gamma_0.99_epochs_80_meta_woCW_resnet18_adam_binary_lamda_1.0/epoch_18.ckpt"
  # strckpt_p2 = "/home/user/model_2022/v220922/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_4by3_244x324.db_221010_LoDzPYZubka3wyJej2JUBA_bsize256_optadam_lr0.0001_gamma_0.99_epochs_80_meta_woCW_resnet18_adam_binary_lamda_1.0/epoch_21.ckpt"
  # strckpt_pdle_p1 = "/home/user/model_2022/v220922_pdle/Train_4C4_SiW_CW_AIHUBx2_CASIA_MSU_OULU_REPLAY_1by1_260x260.db_221014_7Ka6RguFhyzCYsamxtZZw7_bsize128_optadam_lr1e-05_gamma_0.99_epochs_100_meta_woCW_resnet18_adam_pdle_lamda_1.0/epoch_45.ckpt"

  strckpt_p1 = "./ckpts/p1.ckpt"
  strckpt_p2 = "./ckpts/p2.ckpt"
  strckpt_pdle_p1 = "./ckpts/p1_pdle.ckpt"



  ### passed by interface ###
  pilimg, fdbox = loadimgandfd(strtestimg)

  ### Init Color based Engine ####
  kfse = KABANGFASEngine()
  ### Load Models ####
  kfse.loadckpt(strckpt_p1, strckpt_p2, strckpt_pdle_p1)
  ### Gen Patchs according to models ####
  errcode, t_p1, t_p2 = kfse.genpatch2tensor(pilimg, fdbox)
  if errcode == -1:
    print("error code cannot make patches")
  ### Extract scores ####
  kfse.deploy(t_p1[None, :], t_p2[None, :])


if __name__ == '__main__':
  ######################## env
  GPU = 3
  os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(GPU)

  ######################## Test Sample
  strtestimg = "./samples/fake_01.jpg"
  deploycolorfas(strtestimg)

  strtestimg = "./samples/real_01.jpg"
  deploycolorfas(strtestimg)

# patch1 score spoof 0.8523828 vs real 0.1476171
# patch2 score spoof 0.9980133 vs real 0.0019867
# patch1 pdle expected score spoof 0.9935703 vs real 0.0064297
# patch1 score spoof 0.0000638 vs real 0.9999362
# patch2 score spoof 0.0086373 vs real 0.9913628
# patch1 pdle expected score spoof 0.0916376 vs real 0.9083624
import torch.nn as nn

def cross_entropy():
  return nn.CrossEntropyLoss()

def l1_loss():
  return nn.L1Loss()

def mean_squared_loss():
  return nn.MSELoss()

def negative_log_likelihood():
  return nn.CrossEntropyLoss()

def kl_divergence():
  return nn.KLDivLoss()



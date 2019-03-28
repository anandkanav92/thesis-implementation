import torch.optim as optim

def adam_optimizer(model,params):
  return optim.Adam(model.parameters(), params["learning_rate"])



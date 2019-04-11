import torch.optim as optim
from util.constants import Constants


#torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
def adam_optimizer(model,params):
  return optim.Adam(model.parameters(), lr=params[Constants.LEARNING_RATE][Constants.VALUE],eps=params[Constants.EPS][Constants.VALUE],weight_decay=params[Constants.WEIGHT_DECAY][Constants.VALUE])

#torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
def ada_delta(model,params):
  return optim.Adadelta(model.parameters(),lr=params[Constants.LEARNING_RATE][Constants.VALUE],rho=params[Constants.RHO][Constants.VALUE],eps=params[Constants.EPS][Constants.VALUE],weight_decay=params[Constants.WEIGHT_DECAY][Constants.VALUE])

#torch.optim.Adagrad(params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
def ada_grad(model,params):
  return optim.Adagrad(model.parameters(),lr=params[Constants.LEARNING_RATE][Constants.VALUE],lr_decay=params[Constants.LR_DECAY][Constants.VALUE],weight_decay=params[Constants.WEIGHT_DECAY][Constants.VALUE],initial_accumulator_value=params[Constants.INITIAL_ACCUMULATOR_VALUE][Constants.VALUE])

#torch.optim.ASGD(params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
def averaged_sgd(model,params):
  return optim.ASGD(model.parameters(),lr=params[Constants.LEARNING_RATE][Constants.VALUE],weight_decay=params[Constants.WEIGHT_DECAY][Constants.VALUE],alpha=params[Constants.ALPHA][Constants.VALUE],lambd=params[Constants.LAMBD][Constants.VALUE])

#torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
def rms_prop(model,params):
  return optim.RMSprop(model.parameters(), lr=params[Constants.LEARNING_RATE][Constants.VALUE], alpha=params[Constants.ALPHA][Constants.VALUE], eps=params[Constants.EPS][Constants.VALUE], weight_decay=params[Constants.WEIGHT_DECAY][Constants.VALUE], momentum=params[Constants.MOMENTUM][Constants.VALUE])

#torch.optim.SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
def sgd(model,params):
  return optim.SGD(model.parameters(), lr=params[Constants.LEARNING_RATE][Constants.VALUE], momentum=params[Constants.MOMENTUM][Constants.VALUE],  weight_decay=params[Constants.WEIGHT_DECAY][Constants.VALUE])



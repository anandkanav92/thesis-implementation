import time
from mnist_classifier import Black_Magic
from util.postgres_config import connect
from util.backend import *
from util.constants import Constants
from util.json_decoder import Decoder
# server related stuff
from flask import Flask, request
import json
app = Flask(__name__)
from waitress import serve
from collections import namedtuple
import random

import logging
import threading
import time
json_params = {"epoch": [1,5,10,20], "batch_size": [50,100,250,500], "learning_rate": [0.1,0.01,0.0001,.00001], "eps": [0.1,0.01,0.0001,.00001], "weight_decay": [0.1,0.01,0.0001,.00001], "rho": [0.25,0.75,0.5,0.99] , "lr_decay": [0.1,0.01,0.0001,0.00001] , "initial_accumulator_value": [0.1,0.01,0.0001,0.00001] , "alpha": [0.1,0.01,0.0001,0.00001], "lambd": [0.1,0.01,0.0001,0.00001] , "momentum": [0.1,0.01,0.0001,0.00001], "loss_function": [ "cross_entropy","l1_loss","mean_squared_loss","negative_log_likelihood"], "optimizer": [ "adam_optimizer","ada_delta","averaged_sgd","rms_prop","sgd","ada_grad"]}
# json_params = {"epoch": [1,1,1,1,1,1], "batch_size": [50,50,50,50,50,50], "learning_rate": [50,50,50,50,50,50], "eps": [0.1,0.01,0.0001,1,10,5], "weight_decay": [0.1,0.01,0.0001,1,10,5], "rho": [0.1,0.01,0.25,0.5,0.75,0.99] , "lr_decay": [0.1,0.01,0.0001,1,10,5] , "initial_accumulator_value": [0.1,0.01,0.0001,1,10,5] , "alpha": [0.1,0.01,0.0001,1,10,5], "lambd": [0.1,0.01,0.0001,1,10,5] , "momentum": [0.1,0.01,0.0001,1,10,5], "loss_function": [ "cross_entropy","l1_loss","mean_squared_loss","negative_log_likelihood"], "optimizer": [ "sgd","sgd","sgd","sgd","sgd","sgd"]}
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
                    )

def random_text_generator():
  x = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
  return x.join(str(time.time()))
@app.route("/run_model", methods=['GET','POST'])
def startTheModel():
  content = request.get_data()
  my_json = content.decode('utf8').replace("'", '"')
  data_dict = json.loads(my_json,cls=Decoder)
  #data_object = json.loads(my_json, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
  print("recieved params==>")
  print(data_dict)
  print(type(data_dict))
  random_name = random_text_generator()
  logger = logging.getLogger('thread-%s' % random_name)
  logger.setLevel(logging.DEBUG)

  # create a file handler writing to a file named after the thread
  file_handler = logging.FileHandler('thread-%s.log' % random_name)

  formatter = logging.Formatter('(%(threadName)-10s) %(message)s')
  file_handler.setFormatter(formatter)

  logger.addHandler(file_handler)

  threading.Thread(target=main, args=(data_dict,logger)).start()



  # # create a custom formatter and register it for the file handler


  # # register the file handler for the thread-specific logger

  # delay = random.random()
  # t = threading.Thread(target=worker, args=(delay, logger))
  # t.start()
  return '{status: "Success"}'
  #main(data_dict)

def main(params):
  start_time = time.time()
  container = Black_Magic(params)
  #read data
  data_train_loader,data_test_loader = container.read_data()
  print(len(data_test_loader.dataset))
  #train the model
  if container.train(data_train_loader):
    #test the model
    container.predict(data_test_loader)
  else:
    container.precision = -1
  total_time = time.time() - start_time
  cursor = connect()
  if container.precision!=None:
    if insert_precision(cursor,json.dumps(params),container.precision,float(total_time)):
      print("Successfully inserted.")
    else:
      print("Insertion failed.")
  else:
    print("precision is None.")

def set_defaults(params):
  if not Constants.RHO in params:
    params[Constants.RHO] = 0.9
  if not Constants.LR_DECAY in params:
    params[Constants.LR_DECAY] = 0
  if not Constants.INITIAL_ACCUMULATOR_VALUE in params:
    params[Constants.INITIAL_ACCUMULATOR_VALUE] = 0
  if not Constants.LAMBD in params:
    params[Constants.LAMBD] = 0.0001
  if not Constants.ALPHA in params:
    params[Constants.ALPHA] = 0.75
  if not Constants.EPS in params:
    params[Constants.EPS] = 1e-06
  if not Constants.MOMENTUM in params:
    params[Constants.MOMENTUM] = 0
  if not Constants.EPOCH in params:
    params[Constants.EPOCH] = 10
  if not Constants.BATCH_SIZE in params:
    params[Constants.BATCH_SIZE] = 1
  if not Constants.LEARNING_RATE in params:
    params[Constants.LEARNING_RATE] = 1
  if not Constants.WEIGHT_DECAY in params:
    params[Constants.WEIGHT_DECAY] = 1

  return params
def randomize_the_json(key):
  if key == "loss_function":
    return random.randint(0, 3)
  else:
    return random.randint(0, 3)

def set_values(params):
  params["epoch"]["value"] = json_params["epoch"][randomize_the_json("epoch")]
  params["batch_size"]["value"] = json_params["batch_size"][randomize_the_json("batch_size")]
  params["learning_rate"]["value"] = json_params["learning_rate"][randomize_the_json("learning_rate")]
  params["eps"]["value"] = json_params["eps"][randomize_the_json("eps")]
  params["weight_decay"]["value"] = json_params["weight_decay"][randomize_the_json("weight_decay")]
  params["rho"]["value"] = json_params["rho"][randomize_the_json("rho")]
  params["lr_decay"]["value"] = json_params["lr_decay"][randomize_the_json("lr_decay")]
  params["initial_accumulator_value"]["value"] = json_params["initial_accumulator_value"][randomize_the_json("initial_accumulator_value")]
  params["alpha"]["value"] = json_params["alpha"][randomize_the_json("alpha")]
  params["lambd"]["value"] = json_params["lambd"][randomize_the_json("lambd")]
  params["momentum"]["value"] = json_params["momentum"][randomize_the_json("momentum")]
  params["loss_function"]["value"] = json_params["loss_function"][randomize_the_json("loss_function")]
  params["optimizer"]["value"] = json_params["optimizer"][randomize_the_json("optimizer")]
  print(params)
  return params

if __name__ == '__main__':
    #serve(app,host='0.0.0.0', port=5001)

    main_json = {"epoch": {"comments": "", "value": 1.0}, "batch_size": {"comments": "", "value": 100.0}, "learning_rate": {"comments": "", "value": 0.0001}, "eps": {"comments": "", "value": 0.0001}, "weight_decay": {"comments": "", "value": 1e-05}, "rho": {"comments": "", "value": ""}, "lr_decay": {"comments": "", "value": ""}, "initial_accumulator_value": {"comments": "", "value": ""}, "alpha": {"comments": "", "value": 0.01}, "lambd": {"comments": "", "value": ""}, "momentum": {"comments": "", "value": 0.1}, "loss_function": {"comments": "", "value": "negative_log_likelihood"}, "optimizer": {"comments": "", "value": "rms_prop"}}
    # start_time = time.time()
    # param_dummy = {
    #   "user_id" : "random1234",
    #   "background_info" : "voila,I suck at deep learning. But I am gonna kick assin experiments.",
    #   "learning_rate" : 0.01,
    #   "epoch" : 10,
    #   "batch_size" : 5,
    #   # "rho":,
    #   # "lr_decay":,
    #   # "initial_accumulator_value":,
    #   # "lambd":,
    #   # "alpha":,
    #   # "eps":,
    #   # "momentum":,
    #   # "weight_decay":,
    #   "loss_function": "cross_entropy",
    #   # "cross_entropy","l1_loss","mean_squared_loss","negative_log_likelihood","kl_divergence"
    #   "optimizer": "adam_optimizer"
    #   # "adam_optimizer","ada_delta","averaged_sgd","rms_prop","sgd","ada_grad"
    # }
    # params = set_defaults(param_dummy)
    # print(params)
    # main(params)
    #main_json = {'epoch': {'comments': '', 'value': 10}, 'batch_size': {'comments': '', 'value': 500}, 'learning_rate': {'comments': '', 'value': 0.0001}, 'eps': {'comments': '', 'value': 0.0001}, 'weight_decay': {'comments': '', 'value': 0.0001}, 'rho': {'comments': '', 'value': 0.1}, 'lr_decay': {'comments': '', 'value': 0.01}, 'initial_accumulator_value': {'comments': '', 'value': 10}, 'alpha': {'comments': '', 'value': 5}, 'lambd': {'comments': '', 'value': 10}, 'momentum': {'comments': '', 'value': 1}, 'loss_function': {'comments': '', 'value': 'negative_log_likelihood'}, 'optimizer': {'comments': '', 'value': 'sgd'}}
    #main(main_json)
    for i in range(0,50):
      main_json = set_values(main_json)
      main(main_json)
    # #connect to database

    # print("--- %s seconds ---" % (time.time() - start_time))



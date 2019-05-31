import time
from mnist_classifier import Black_Magic
from util.postgres_config import connect
from util.backend import *
from util.constants import Constants
from util.json_decoder import *
# server related stuff
from flask import Flask, request
import json
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
from multiprocessing import Process

from waitress import serve
from collections import namedtuple
import random
from imagenette import Imagenette
import logging
import time
import torchvision.transforms as transforms
import uuid
import string
from memory_profiler import profile
training_process = None

def random_text_generator():
  x = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))
  return x.join(str(time.time()))

random_name = random_text_generator()
logger = logging.getLogger('thread-%s' % random_name)
logger.setLevel(logging.DEBUG)

# json_params = {"epoch": [10,20,30,35], "batch_size": [5,10,25,50], "learning_rate": [0.1,0.01,0.0001,.00001], "eps": [0.1,0.01,0.0001,.00001], "weight_decay": [0.1,0.01,0.0001,.00001], "rho": [0.25,0.75,0.5,0.99] , "lr_decay": [0.1,0.01,0.0001,0.00001] , "initial_accumulator_value": [0.1,0.01,0.0001,0.00001] , "alpha": [0.1,0.01,0.0001,0.00001], "lambd": [0.1,0.01,0.0001,0.00001] , "momentum": [0.1,0.01,0.0001,0.00001], "loss_function": [ "cross_entropy","l1_loss","mean_squared_loss","negative_log_likelihood"], "optimizer": [ "adam_optimizer","ada_delta","averaged_sgd","rms_prop","sgd","ada_grad"]}


# json_params = {"epoch": [1,1,1,1,1,1], "batch_size": [50,50,50,50,50,50], "learning_rate": [50,50,50,50,50,50], "eps": [0.1,0.01,0.0001,1,10,5], "weight_decay": [0.1,0.01,0.0001,1,10,5], "rho": [0.1,0.01,0.25,0.5,0.75,0.99] , "lr_decay": [0.1,0.01,0.0001,1,10,5] , "initial_accumulator_value": [0.1,0.01,0.0001,1,10,5] , "alpha": [0.1,0.01,0.0001,1,10,5], "lambd": [0.1,0.01,0.0001,1,10,5] , "momentum": [0.1,0.01,0.0001,1,10,5], "loss_function": [ "cross_entropy","l1_loss","mean_squared_loss","negative_log_likelihood"], "optimizer": [ "sgd","sgd","sgd","sgd","sgd","sgd"]}


logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
                    )

@app.route("/save_backgroundinfo", methods=['GET','POST'])
def save_background_info():
  content = request.get_data()
  my_json = content.decode('utf8').replace("'",'"')
  data_dict = json.loads(my_json,cls=Decoder_int)
  cursor = connect(logger)
  user_ids = fetch_all_users(cursor,logger)
  found = False;
  user_id=''
  while not found:
    user_id = uuid.uuid4().hex[:6].upper()
    if (user_id not in user_ids) and user_id is not '':
      found=True
  data_dict['user_id'] = user_id
  result = insert_background_info(cursor,data_dict,logger)
  logger.debug(json.dumps(result))
  return json.dumps(result)


@app.route("/get_results", methods=['GET'])
def get_user_results():
  data_dict = json.loads(json.dumps(request.args),cls=Decoder_int)
  cursor = connect(logger)
  results = get_result_data(cursor,data_dict['user_id'],logger)
  #logger.debug(results)
  if results is None:
    return json.dumps({"status": 0})
  else:
    logger.debug(results)
    return json.dumps({"status": 1,"rows" : results})


@app.route("/run_model", methods=['GET','POST'])
def startTheModel():
  global training_process
  content = request.get_data()
  my_json = content.decode('utf8').replace("'", '"')
  data_dict = json.loads(my_json,cls=Decoder)


  # create a file handler writing to a file named after the thread
  file_handler = logging.FileHandler('thread-%s.log' % random_name)

  formatter = logging.Formatter('(%(threadName)-10s) %(message)s')
  file_handler.setFormatter(formatter)

  logger.addHandler(file_handler)

  logger.debug("data_dict['finalSubmission']['value'] {} and training_process {}".format(type(data_dict['finalSubmission']['value']),training_process))
  if data_dict['finalSubmission']['value'] == True and training_process is None:
    cursor = connect(logger)
    if cursor is not None:
      if insert_user_results_final(cursor,json.dumps(data_dict),logger,data_dict['user_id']):
        logger.info("Successfully inserted final record for {}".format(data_dict['user_id']))
        return json.dumps({'status': 0}) #final record inserted

      else:
        logger.info("Insertion failed for final record {}".format(data_dict['user_id']))
        return json.dumps({'status': 1}) #final record insertion failed



  if training_process is None or (not training_process.is_alive()):
    data_dict = set_defaults(data_dict)
    cursor = connect(logger)
    #to handle cases where the training is killed intermediately
    row_id = insert_user_results(cursor,json.dumps(data_dict),-2,-1,logger,data_dict['user_id'])
    if row_id is not None:
      data_dict['row_id'] = row_id
    else:
      logger.debug("Insert failed at the start!")
    training_process = Process(target=main, args=(data_dict,logger), name=data_dict['user_id']+random_text_generator())
    training_process.start()

  #threading.Thread(target=main, args=(data_dict,logger), name=data_dict['user_id']).start()
  return json.dumps({'status': 2}) #depeicts training has started

@app.route("/get_status", methods=['GET'])
def get_execution_status():
  global training_process
  data_dict = json.loads(json.dumps(request.args),cls=Decoder_int)
  logger.debug("data is here:{}".format(data_dict))
  if training_process is not None:
    if training_process.is_alive():
      logger.debug(json.dumps({'status': 0}))
      return json.dumps({'status': 0}) #ongiong
  logger.debug(json.dumps({'status': 1}))
  return json.dumps({'status': 1}) #finished


@app.route("/cancel_job", methods=['GET'])
def cancel_model_training():
  global training_process
  data_dict = json.loads(json.dumps(request.args),cls=Decoder_int)
  if training_process is not None:
    training_process.terminate()
    training_process.join()
    time.sleep(2)
    if not training_process.is_alive():
      logger.debug(json.dumps({'status': 1}))
      logger.debug(training_process.exitcode)
      return json.dumps(json.dumps({'status': 1})) #means still killed
  logger.debug(training_process)
  return json.dumps(json.dumps({'status': 0})) #means running


def main(params,logger):

  start_time = time.time()

  #set defaults
  # params = set_defaults(params)

  container = Black_Magic(params)
  logger.info("recieved parameters: ")
  logger.info(json.dumps(params))
  #read data
  # data_train_loader,data_test_loader = container.read_data_mnist()
  data_train_loader,data_test_loader = container.read_data_imagenette()
  # container.read_fastai_imagenette()
  #train the model
  precision = None
  if container.train(data_train_loader):
    #test the model
    precision = container.predict(data_test_loader)
  else:
    container.precision = -1
    precision = -1
  total_time = time.time() - start_time
  logger.info("training and testing finished.")
  cursor = connect(logger)
  if container.precision is not None:
    if cursor is not None:
      if params['row_id'] is not None:
        if update_user_results(cursor,container.precision,float(total_time),logger,params['row_id']):
          logger.info("Successfully inserted.")
      else:
        logger.info("update failed.")
    else:
      logger.info(json.dumps(params)+" "+str(container.precision)+" "+str(float(total_time)))
  else:
    logger.critical("precision is None.")




def set_defaults(params):
  if params[Constants.RHO][Constants.VALUE] == '':
    params[Constants.RHO][Constants.VALUE] = 0.9
  if params[Constants.LR_DECAY][Constants.VALUE] == '':
    params[Constants.LR_DECAY][Constants.VALUE] = 0
  if params[Constants.INITIAL_ACCUMULATOR_VALUE][Constants.VALUE] == '':
    params[Constants.INITIAL_ACCUMULATOR_VALUE][Constants.VALUE] = 0
  if params[Constants.LAMBD][Constants.VALUE] == '':
    params[Constants.LAMBD][Constants.VALUE] = 0.0001
  if params[Constants.ALPHA][Constants.VALUE] == '':
    params[Constants.ALPHA][Constants.VALUE] = 0.75
  if params[Constants.EPS][Constants.VALUE] == '':
    params[Constants.EPS][Constants.VALUE] = 1e-06
  if params[Constants.MOMENTUM][Constants.VALUE] == '':
    params[Constants.MOMENTUM][Constants.VALUE] = 0
  if params[Constants.EPOCH][Constants.VALUE] == '':
    params[Constants.EPOCH][Constants.VALUE] = 10
  if params[Constants.BATCH_SIZE][Constants.VALUE] == '':
    params[Constants.BATCH_SIZE][Constants.VALUE] = 1
  if params[Constants.LEARNING_RATE][Constants.VALUE] == '':
    params[Constants.LEARNING_RATE][Constants.VALUE] = 1
  if params[Constants.WEIGHT_DECAY][Constants.VALUE] == '':
    params[Constants.WEIGHT_DECAY][Constants.VALUE] = 1
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
  # serve(app,host='0.0.0.0', port=5001)

  params = {"epochs": {"value": 105, "comment": ""}, "batchSize": {"value": 100.0, "comment": ""}, "lossFunction": {"value": "cross_entropy", "comment": ""}, "optimizer": {"value": "adam_optimizer", "comment": ""}, "learningRate": {"value": 0.0001, "comment": ""}, "epsilon": {"value": 1e-05, "comment": ""}, "weightDecay": {"value": 0.001, "comment": ""}, "rho": {"value": 0.9, "comment": ""}, "learningRateDecay": {"value": 0, "comment": ""}, "initialAccumulator": {"value": 0.1, "comment": ""}, "alpha": {"value": 0, "comment": ""}, "lambda": {"value": 0.01, "comment": ""}, "momentum": {"value": 0.9, "comment": ""}, "user_id": "F88BC8"}
  main(params,logger)

 # main_json = {"epoch": {"comments": "", "value": 50.0}, "batch_size": {"comments": "", "value": 1}, "learning_rate": {"comments": "", "value": 0.0001}, "eps": {"comments": "", "value": 0.0001}, "weight_decay": {"comments": "", "value": 1e-05}, "rho": {"comments": "", "value": ""}, "lr_decay": {"comments": "", "value": ""}, "initial_accumulator_value": {"comments": "", "value": ""}, "alpha": {"comments": "", "value": 0.01}, "lambd": {"comments": "", "value": ""}, "momentum": {"comments": "", "value": 0.1}, "loss_function": {"comments": "", "value": "negative_log_likelihood"}, "optimizer": {"comments": "", "value": "adam_optimizer"}}
  # # main(main_json)

  # for i in range(0,50):
  #   main_json = set_values(main_json)
  #   main(main_json)
  # transform=transforms.Compose([
  #                          transforms.Resize((32, 32)),
  #                          transforms.ToTensor()])

  # imagenette_set = Imagenette( '/Users/kanavanand/Downloads/imagenette-160/',transform=transform,target_transform=transform)



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
    # main(params)
    #main_json = {'epoch': {'comments': '', 'value': 10}, 'batch_size': {'comments': '', 'value': 500}, 'learning_rate': {'comments': '', 'value': 0.0001}, 'eps': {'comments': '', 'value': 0.0001}, 'weight_decay': {'comments': '', 'value': 0.0001}, 'rho': {'comments': '', 'value': 0.1}, 'lr_decay': {'comments': '', 'value': 0.01}, 'initial_accumulator_value': {'comments': '', 'value': 10}, 'alpha': {'comments': '', 'value': 5}, 'lambd': {'comments': '', 'value': 10}, 'momentum': {'comments': '', 'value': 1}, 'loss_function': {'comments': '', 'value': 'negative_log_likelihood'}, 'optimizer': {'comments': '', 'value': 'sgd'}}
    #main(main_json)

    # #connect to database


#import sys
#main(sys.argv[1])


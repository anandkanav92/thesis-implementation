import time
from mnist_classifier import Black_Magic
from util.postgres_config import connect
def main(params):
  container = Black_Magic(params)
  #read data
  data_train_loader,data_test_loader = container.read_data()
  print(len(data_test_loader.dataset))
  #train the model
  container.train(data_train_loader)
  #test the model
  container.predict(data_test_loader)



if __name__ == '__main__':
    start_time = time.time()
    # param_dummy = {
    #   "learning_rate" : 0.01,
    #   "epoch" : 10,
    #   "batch_size" : 5,
    #   "loss_function": "cross_entropy",
    #   "optimizer": "adam_optimizer"
    # }
    # main(param_dummy)

    connect()
    #connect to database

    print("--- %s seconds ---" % (time.time() - start_time))


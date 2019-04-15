
import json
import os

def run_job(params):
  run_job_cmd = " sbatch run_script.sh '"+params+"'"
  print(run_job_cmd)
  os.system(run_job_cmd)

if __name__ == '__main__':
  main_json = {"epoch": {"comments": "", "value": 1.0}, "batch_size": {"comments": "", "value": 100.0}, "learning_rate": {"comments": "", "value": 0.0001}, "eps": {"comments": "", "value": 0.0001}, "weight_decay": {"comments": "", "value": 1e-05}, "rho": {"comments": "", "value": ""}, "lr_decay": {"comments": "", "value": ""}, "initial_accumulator_value": {"comments": "", "value": ""}, "alpha": {"comments": "", "value": 0.01}, "lambd": {"comments": "", "value": ""}, "momentum": {"comments": "", "value": 0.1}, "loss_function": {"comments": "", "value": "negative_log_likelihood"}, "optimizer": {"comments": "", "value": "rms_prop"}}
  params = json.dumps(main_json)
  print(params)
  run_job(params)

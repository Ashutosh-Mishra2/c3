#!/usr/bin/env python
"""Base run for c3 code."""
import logging
logging.getLogger('tensorflow').disabled = True
import os
import shutil
import json
import pickle
import argparse
import c3po.utils.parsers as parsers
import c3po.utils.tf_utils as tf_utils
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument("master_config")
args = parser.parse_args()
master_config = args.master_config
with open(master_config, "r") as cfg_file:
    cfg = json.loads(cfg_file.read())
optim_type = cfg['optim_type']
exp_setup = cfg['exp_setup']
opt_config = cfg['optimizer_config']

tf_utils.tf_setup()
with tf.device('/CPU:0'):
    exp = parsers.create_experiment(exp_setup)

    if optim_type == "C1":
        opt = parsers.create_c1_opt(opt_config, exp.model.lindbladian)
    elif optim_type == "C2":
        eval_func = cfg['eval_func']
        opt, exp_right = parsers.create_c2_opt(opt_config, eval_func)
    elif optim_type == "C3":
        opt = parsers.create_c3_opt(opt_config)
    else:
        raise Exception("C3:ERROR:Unknown optimization type specified.")
    opt.set_exp(exp)
    dir = opt.logdir

    shutil.copy2(master_config, dir)
    shutil.copy2(exp_setup, dir)
    shutil.copy2(opt_config, dir)
    if optim_type == "C2":
        shutil.copy2(eval_func, dir)

    if 'initial_point' in cfg:
        try:
            init_point = cfg['initial_point']
            opt.load_best(init_point)
            print(
                "C3:STATUS:Loading initial point from : "
                f"{os.path.abspath(init_point)}"
            )
            shutil.copy(init_point, dir+"initial_point.log")
        except FileNotFoundError:
            print(
                f"C3:STATUS:No initial point found at "
                f"{os.path.abspath(init_point)}. "
                "Continuing with default."
            )

    if 'real_params' in cfg:
        real_params = cfg['real_params']

    if optim_type == "C1":
        opt.optimize_controls()

    elif optim_type == "C2":
        real = {'params': [
            par.numpy().tolist()
            for par in exp_right.get_parameters()]
        }
        with open(dir + "real_model_params.log", 'w') as real_file:
            real_file.write(json.dumps(exp_right.id_list))
            real_file.write("\n")
            real_file.write(json.dumps(real))
            real_file.write("\n")
            real_file.write(exp_right.print_parameters())

        opt.optimize_controls()

    elif optim_type == "C3":
        learn_from = []
        opt.read_data(cfg['datafile'])
        shutil.copy2(
            "/".join(cfg['datafile']['left'].split("/")[0:-1]) \
            + "/real_model_params.log",
            dir
        )
        opt.learn_model()

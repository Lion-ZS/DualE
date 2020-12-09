import config
from  models import *
import json
import os 
con = config.Config()
con.set_in_path("./benchmarks/WN18RR/")
con.set_work_threads(8)
con.set_train_times(20000)
con.set_nbatches(10)
con.set_alpha(0.022)
con.set_bern(1)
con.set_dimension(200)#200
con.set_lmbda(0.25)
con.set_lmbda_two(0.25)
con.set_margin(1.0)
con.set_ent_neg_rate(2)
con.set_rel_neg_rate(0)
con.set_opt_method("adagrad")
con.set_save_steps(20000)
con.set_valid_steps(20000)
con.set_early_stopping_patience(10)
con.set_checkpoint_dir("./checkpoint")
con.set_result_dir("./result")
con.set_test_link(True)
con.set_test_triple(True)
con.init()
con.set_train_model(DualE)
con.train()


#!/usr/bin/bash 

(nohup python3 nn_train.py --config config_train/train_evc_cfg.json --device cuda:0 && 
 nohup python3 nn_test.py --config config_test/test_evc_config.json --device cuda:0) & 
#!/bin/bash
# Copyright (c) 2016 Emotibot. All Rights Reserved

#set -e

# run inference 
python srl_inference.py

# run script
perl out/conlleval.pl < out/eval_test.txt > out/res_temp.txt
cat out/res_temp.txt
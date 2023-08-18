#!/bin/bash
cd /dls/app/trtr
torchrun
        --standalone
        --nproc_per_node=8
        main.py
        --d_model=1024
        --batch_size=1024
        --train_epochs=0
        --index_path=/dls/app/trtr/data/index.dat
        --data_path=/dls/app/trtr/data/data.dat

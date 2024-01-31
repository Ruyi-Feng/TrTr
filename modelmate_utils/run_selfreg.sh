#!/bin/bash
cd /dls/app/trtr
echo "============== start training ================="
python -m torch.distributed.launch --use_env --nproc_per_node 8 main.py --used_yml /dls/app/trtr/script/selfregression.yml --train_epoch 0
echo "finish init weights"
cd ..
mkdir model_reg
cp ./trtr/checkpoints/pretrain/checkpoint_cpu_last.pth ./model_reg/checkpoint_cpu_last.pth
ls
echo "-------------start regest init---------------"
python regest_last.py --tags selfreg

cd /dls/app/trtr
python -m torch.distributed.launch --use_env --nproc_per_node 8 main.py --used_yml /dls/app/trtr/script/selfregression.yml --train_epoch 50
echo "============== finish training ================="
cd ./checkpoints/pretrain/
ls
python regest_best.py --tags selfreg
echo "regest best finish!!"
sleep 10000

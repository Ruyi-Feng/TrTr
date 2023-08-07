# TrTr

## Project structure

- trtr
  - net
- pretrain
  - exp
  - mask
  - dataset
  - param
  - data_provider
- finetune
  - exp
  - finetune_tasks
  - dataset
  - data_factory
  - param
- checkpoints
  - pretrain
  - finetune
    - task1
    - task2
    - task3
- data
  - train
  - val


## TODO:

1. random.sample 没改 [done]
2. compensation作为pretrain [done]
3. frm_mark做position_embed [-]
4. relative position embed（net）[done]
5. 加长序列 & 数据集制作
6. 去掉switch，在T5的answer模式下它学不到序列信息，会误导。车辆不会倒退 [done]


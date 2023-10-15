import argparse

def params():
    parser = argparse.ArgumentParser(description='transformer parameters')
    parser.add_argument('--task', type=str, default='pretrain', help='pretrain or sepecific finetune tasks')
    parser.add_argument('--save_path', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--sepecific', type=str, default='./checkpoints/cpu_histreg_d512_in120_pd60_loss0.092_epoch100.pth', help='location of sepecific model checkpoints')
    parser.add_argument('--index_path', type=str, default='./data/train/index.bin')
    parser.add_argument('--data_path', type=str, default='./data/train/data.bin')

    parser.add_argument('--is_train', type=bool, default=True, help='if True is train model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='original leaning rate')
    parser.add_argument('--train_epochs', type=int, default=100, help='total train epoch')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--drop_last', type=bool, default=True)
    parser.add_argument('--max_car_num', type=int, default=10, help='max car num in a frame')
    parser.add_argument('--input_len', type=int, default=120, help='')  # 前input_len 个数据 如果是用hist-reg则input和pred不一样
    parser.add_argument('--pred_len', type=int, default=60, help='')
    parser.add_argument('--shared_pos_embed', type=bool, default=True, help='')

    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=6, help='num of decoder layers')
    parser.add_argument('--model_type', type=str, default='nrml', help='rltv, nrml, gpt')

    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--lradj', default='type1')
    args = parser.parse_args()

    return args


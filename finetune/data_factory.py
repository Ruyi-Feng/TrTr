from finetune.finetune_tasks import Data_Compensation, Data_Summary, Data_Control

data_dict = {
    'compensation': Data_Compensation,
    'summary': Data_Summary,
    'control': Data_Control,
}

def gen_dataset(config):
    mark = config.task
    Data = data_dict[mark]
    data_set = Data(index_path=config.index_path,
                    data_path=config.data_path,
                    max_car_num=config.max_car_num,
                    input_len=config.input_len,
                    pred_len=config.pred_len)
    return data_set

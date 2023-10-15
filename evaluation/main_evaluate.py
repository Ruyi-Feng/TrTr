from evaluation.evaluation import Evaluation
import os


if __name__ == '__main__':
    # os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    # local_rank = int(os.environ['LOCAL_RANK'])

    # for tsk in ["pretrain"]:  # , 
    #     rslt_pth = "./results/rslt-10000.json"
    #     eva = Evaluation(if_pretrain=True, task=tsk)
    #     eva.obtain_results(rslt_path=rslt_pth, local_rank=local_rank)

    for tsk in ["compensation", ]:
        rslt_pth = "./results/"+"rslt-10000.json"
        eva = Evaluation(if_pretrain=True, task=tsk)
        print("----------start check results-----------")
        eva.check_results(check_list=["iou"], rslt_path=rslt_pth)  # "overlap", "rmse", "spd_stable", "DCN", "iou"




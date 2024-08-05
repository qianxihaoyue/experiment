import os
from pathlib import Path
import torch
from torch.nn import functional as F



def create_paths(result_base_path):
    result_path = create_experiment_name(result_base_path=result_base_path,mode="number")
    for p in ["log","view","predict","checkpoints","test"]:
        sub_path=os.path.join(result_path,p)
        Path(sub_path).mkdir(parents=True,exist_ok=True)

    return result_path



def create_experiment_name(result_base_path,mode="number"):
    for i in range(1,999):
        result_path=os.path.join(result_base_path,f"exp_{i}")
        if not os.path.exists(result_path):
            Path(result_path).mkdir(parents=True, exist_ok=True)
            return  result_path


def create_run_name(result_base_path,mode="number"):
    for i in range(1,999):
        result_path=os.path.join(result_base_path,f"run_{i}")
        if not os.path.exists(result_path):
            Path(result_path).mkdir(parents=True, exist_ok=True)
            return  result_path
# def find_experiment_name(result_base_path):
#     list_paths=os.listdir(result_base_path)
#     num_list=[]
#     for path in list_paths:
#         num_list.append(path.split("_")[1])
#     num=max(num_list)
#     return os.path.join(result_base_path,f"exp_{num}")


def show_args(opt,logger):
    dict=vars(opt)
    for key,value in dict.items():
        logger.info(f"{key}: {value}")





def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()



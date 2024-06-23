import os
from pathlib import Path





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


#
# def find_experiment_name(result_base_path):
#     list_paths=os.listdir(result_base_path)
#     num_list=[]
#     for path in list_paths:
#         num_list.append(path.split("_")[1])
#     num=max(num_list)
#     return os.path.join(result_base_path,f"exp_{num}")

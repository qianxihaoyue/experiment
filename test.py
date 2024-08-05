from safetensors import torch
from utils.metric import calc_dice
from options.test_options import TestOptions
from torch.utils.data import DataLoader
from utils.commonDataset import CommonDataset
from models.unet.unet import UNet
import numpy as np
from monai.metrics import DiceMetric
import torch
from utils.metric import calc_dice
import os
from pathlib import Path
from utils.tool import *
from utils.logger import  *
from utils.visualizer import *
@torch.no_grad()
def test(model,opt, test_loader,test_dataset):
    dice_list=[]
    for i,batch in enumerate(test_loader):
        image=batch["image"]
        label=batch["label"].numpy()
        images=image.to(opt.device)
        output=model(images)
        output=output.sigmoid()
        output=output.detach().cpu().numpy().squeeze(1)
        output=(output>=0.5).astype(np.uint8)
        dices=calc_dice(output,label)
        dice_list.extend(dices)

    return {"mdice":sum(dice_list)/len(dice_list)}




if __name__ == '__main__':
    opt=TestOptions().get_opts()
    opt.result_path = os.path.join(os.path.dirname(os.path.dirname(opt.trained_checkpoint)), "test",os.path.basename(opt.trained_checkpoint).split(".")[0])
    opt.result_path = create_run_name(opt.result_path)
    logger = Logger(os.path.join(opt.result_path, "test_log.txt")).get_logger()
    show_args(opt, logger)
    opt.result_path = os.path.join(opt.result_path, "save")    #当前路径存放可视化图片，注意这段代码的先后顺序
    Path(opt.result_path).mkdir(exist_ok=True, parents=True)


    model=UNet()
    model.load_state_dict(torch.load(opt.trained_checkpoint))
    model.cuda()
    model.eval()

    test_dataset = CommonDataset(opt.test_root_path, opt.subdir, opt.image_size)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    visualizer = Visualizer()

    metircs=test(model,opt,test_dataloader,test_dataset)

    logger.info(f'mdice:{metircs["mdice"]}')

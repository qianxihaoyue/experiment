from safetensors import torch
from utils.metric import calc_dice
from options.test_options import TestOptions
from torch.utils.data import DataLoader
from prepare.commonDataset import CommonDataset
from models.myunet.unet import Unet
import numpy as np
from monai.metrics import DiceMetric
import torch
from utils.metric import calc_dice
import os
from pathlib import Path
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
    model=Unet()
    model.load_state_dict(torch.load(opt.trained_checkpoint))
    model.cuda()
    model.eval()
    test_dataset = CommonDataset(opt.test_root_path, opt.subdir, opt.image_size)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)
    metircs=test(model,opt,test_dataloader,test_dataset)
    print(metircs["mdice"])

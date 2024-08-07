import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import random
import warnings
warnings.filterwarnings("ignore")
import argparse
import os
import sys
from utils.tool import *
from models.unet import UNet
from models.unet2 import UNet2
from models.utransformer import UTransformer
from models.resunet import ResUNet
from options.train_options import TrainOptions
from utils.commonDataset import  CommonDataset
from torch.nn import BCEWithLogitsLoss
from test import test
from utils.logger import Logger
def train(model,opt,train_dataloader,train_dataset,test_dataloader,test_dataset,criterion,optimizer,logger):
    for epoch in range(1,opt.epochs+1):
        epoch_loss = 0
        for i, batch in enumerate(train_dataloader):
            images=batch['image']
            labels=batch['label']
            optimizer.zero_grad()
            images=images.to(opt.device)
            labels=labels.to(opt.device)
            outputs = model(images)
            loss = structure_loss(outputs,labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        metrics=test(model,opt,test_dataloader,test_dataset)
        logger.info(f"Epoch {epoch}/{opt.epochs}, Loss: {epoch_loss},mdice: {metrics['mdice']}")
        if epoch % opt.checkpoint_save_freq == 0:
            torch.save(model.state_dict(), f"{opt.result_path}/checkpoints/epoch_{epoch}.pth")




if __name__ == '__main__':
    opt = TrainOptions().get_opts()
    opt.result_path=create_paths(opt.result_path)
    logger=Logger(f"{opt.result_path}/log/train_log.txt").get_logger()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    model=ResUNet(in_channels=opt.in_channels,n_classes=opt.n_classes)
    model.cuda()
    model.train()

    train_dataset=CommonDataset(opt.root_path,opt.subdir,opt.image_size)
    train_dataloader=DataLoader(train_dataset,batch_size=opt.batch_size,shuffle=True)
    test_dataset=CommonDataset(opt.test_root_path,opt.subdir,opt.image_size)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    criterion=BCEWithLogitsLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=opt.lr,weight_decay=opt.weight_decay)

    train(model,opt,train_dataloader,train_dataset,test_dataloader,test_dataset,criterion,optimizer,logger)

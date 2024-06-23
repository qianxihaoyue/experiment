from pathlib import  Path
import os
import  torch.nn.functional as F
from monai.data.ultrasound_confidence_map import cv2
import numpy as np
from options.predict_options import PredictOptions
from models.myunet.unet import  Unet
import torch
import cv2
import datetime
from torchvision import transforms
from tqdm import tqdm
def predict(opt, model):
    sample_list=os.listdir(opt.image_folder)
    for sample in tqdm(sample_list):
        img=cv2.imread(os.path.join(opt.image_folder,sample))
        h,w=img.shape[:2]
        trans=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((opt.image_size, opt.image_size)),
            transforms.ToTensor(),
        ])
        img_new=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img_new=trans(img_new).unsqueeze(0).cuda()
        predict=model(img_new)
        predict=F.interpolate(predict,size=(h,w),mode='bilinear',align_corners=True)
        predict=predict.detach().squeeze().cpu()
        predict=predict.sigmoid()
        predict=(predict-predict.min())/(predict.max()-predict.min())
        predict=predict.numpy()
        predict=(predict>=0.5).astype(np.uint8)*255
        predict=predict[...,None]
        predict=np.repeat(predict,3,axis=2)
        merged=np.concatenate([img,predict],axis=1)
        cv2.imwrite(os.path.join(opt.result_path,sample),merged)







if __name__=='__main__':
    opt=PredictOptions().get_opts()
    model=Unet()
    model.load_state_dict(torch.load(opt.trained_checkpoint))
    model.cuda()
    model.eval()

    opt.result_path =os.path.join(os.path.dirname(os.path.dirname(opt.trained_checkpoint)), "predict", os.path.basename(opt.trained_checkpoint).split(".")[0],datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    Path(opt.result_path).mkdir(exist_ok=True, parents=True)
    predict(opt, model)
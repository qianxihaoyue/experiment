from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
class CommonDataset(Dataset):
    def __init__(self,root_path,subdir,image_size):
        self.image_size = image_size
        self.subdir = subdir
        self.root_path=root_path

        self.sample_list=os.listdir(os.path.join(self.root_path,self.subdir[0]))
        self.image_paths=[os.path.join(self.root_path,self.subdir[0],sample_name) for sample_name in  self.sample_list]
        self.label_paths=[os.path.join(self.root_path,self.subdir[1],sample_name) for sample_name in self.sample_list]

        self.transform=transforms.Compose([
            transforms.Resize((self.image_size,self.image_size)),
            transforms.ToTensor()])

        self.transform_seg=transforms.Compose([
            transforms.Resize((self.image_size,self.image_size)),
            transforms.ToTensor()])


    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, idx):
        name=self.sample_list[idx]
        image=Image.open(os.path.join(self.root_path,self.subdir[0],name)).convert('RGB')
        image_torch=self.transform(image)
        label=Image.open(os.path.join(self.root_path,self.subdir[1],name)).convert('L')
        label_torch=self.transform_seg(label)

        return { 'image': image_torch, 'label': label_torch }
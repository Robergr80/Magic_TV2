import numpy as np
import torch
from torch.utils.data import Dataset
from einops import repeat, rearrange
from torchvision import utils
import random
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop
import torchvision.transforms as transforms
from tqdm import tqdm
from model.sampling_retina_3 import *

def area_std(labels,cor_ls):
#     labels = labels.numpy()
    dic = {}
    for i in range(len(labels)):
        if labels[i] not in dic:
            dic[labels[i]] = 0
#         pixel = label_to_color[labels[i]]
        x_off,y_off,kernel_radius = cor_ls[i]
        kernel_radius_int = max(int(kernel_radius),1)
        dic[labels[i]]+=kernel_radius_int**2
    return np.std(list(dic.values()))

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, targets, transform=None):
#         assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.targets = targets
        self.transform = transform
    def __getitem__(self, index):
        x = self.tensors[index]
        if self.transform:
            x = self.transform(x)
        y = self.targets[index]
        return x, y
    def __len__(self):
        
        return len(self.tensors)
    
def take_indexes_channel_perm(sequences, indexes):
    return torch.gather(sequences, -2, repeat(indexes, 't f ->b t f c',b = sequences.shape[0],c=3))

def visFilters(weights, nrow=24, patch_size=4, padding=1): 
    n = weights.size(0)
    filters = weights.reshape(n, patch_size, patch_size, 3).permute(0,3,1,2)
    rows = np.min((filters.shape[0] // nrow + 1, 64))
#     print(filters.shape,row)
    grid = utils.make_grid(filters, nrow=nrow, normalize=True, padding=padding)
    return grid
# .permute((1, 2, 0))
# train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=ToTensor(),)

def unflatten_index(n,cols=32):
    return n//cols,n%cols


def take_indexes_channel_perm_global(sequences, indexes):
#     return torch.gather(sequences, -2, repeat(indexes, 't f ->b t f c',b = sequences.shape[0],c=3))
    b,c,_ =sequences.shape
    return torch.gather(sequences, -1, repeat(indexes,'f -> b c f',b=b,c=c))

class GlobalPerm(object):

    def __init__(self,idx):
        self.image_size = int(np.sqrt(len(idx)))
        self.forward_perm =idx
        self.backward_perm = torch.argsort(self.forward_perm)
    
    def __call__(self, img):
        val_img_flatten = rearrange(img,"a b c d -> a b (c d)")
        val_img_flatten_permuted = take_indexes_channel_perm_global(val_img_flatten,self.forward_perm.to(val_img_flatten.device))
        val_img_re = rearrange(val_img_flatten_permuted,"a b (c d) -> a b c d", c = self.image_size)
        return val_img_re
    
    def inv(self,img):
        val_img_flatten = rearrange(img,"a b c d -> a b (c d)")
        val_img_flatten_permuted = take_indexes_channel_perm_global(val_img_flatten,self.backward_perm.to(val_img_flatten.device))
        val_img_re = rearrange(val_img_flatten_permuted,"a b (c d) -> a b c d", c = self.image_size)
        return val_img_re
    
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    
def load_cifar_in_memory(kernel_pyramid,load_batch_size=200,data_crop_rate=0.85,device = "cuda"):
    train_dataset_eval = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), RandomResizedCrop(32, scale=(data_crop_rate, data_crop_rate), ratio=(1.0, 1.0)), Normalize(0.5, 0.5),Upsample_transform(upsize_ratio)]))
    train_dl = torch.utils.data.DataLoader(train_dataset_eval, batch_size=load_batch_size, shuffle=False, num_workers=4,pin_memory=True,drop_last=True)
    val_dataset_eval = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), RandomResizedCrop(32, scale=(data_crop_rate, data_crop_rate), ratio=(1.0, 1.0)) ,Normalize(0.5, 0.5),Upsample_transform(upsize_ratio)]))
    test_dl = torch.utils.data.DataLoader(val_dataset_eval, batch_size=load_batch_size, shuffle=False, num_workers=4,pin_memory=True,drop_last=True)
    data_train = []
    train_label = []
    kernel_pyramid=kernel_pyramid.to(device)
    for img, label in tqdm(iter(train_dl)):
        img = img.to(device)
        x = sample(img,kernel_pyramid,mask)
        data_train.append(x.cpu())
        train_label.append(label)
    data_train = torch.cat(data_train)
    train_label = torch.cat(train_label)

    data_test = []
    test_label = []
    for img, label in tqdm(iter(test_dl)):
        img = img.to(device)
        x = sample(img,kernel_pyramid,mask)
        data_test.append(x.cpu())
        test_label.append(label)
    data_test = torch.cat(data_test)
    test_label = torch.cat(test_label)
    
    
#     data_train = torch.load("tmp/tmp_data/data_train.pt")
#     data_test = torch.load("tmp/tmp_data/data_test.pt")
#     train_label = torch.load("tmp/tmp_data/train_label.pt")
#     test_label = torch.load("tmp/tmp_data/test_label.pt")
    return data_train,data_test,train_label,test_label
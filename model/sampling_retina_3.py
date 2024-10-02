from sklearn.manifold import spectral_embedding
import numpy as np
import scipy
import torch
import numpy as np
# import cv2
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.functional import conv2d
import math
import torchvision.transforms as transforms
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange

def calc_MI(x, y, bins):
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi
def unflatten_index(n,cols=32):
    return n//cols,n%cols

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
def imshow(img):
    plt.figure(figsize = (10,10))
    plt.imshow(img.permute(1,2,0).numpy())
    plt.axis('off')
    plt.show()
    

class Upsample_transform(object):
    def __init__(self,scale_ratio =2):
        self.m = nn.Upsample(scale_factor=scale_ratio, mode = 'nearest')
    def __call__(self, pic):
        return self.m(pic.unsqueeze(0)).squeeze(0)
    def __repr__(self):
        return self.__class__.__name__ + '()'

def gaussian_kernel(size=5, device=torch.device('cpu'), channels=3, sigma=1, dtype=torch.float):
    # Create Gaussian Kernel. In Numpy
    interval  = (2*sigma +1)/(size)
    ax = np.linspace(-(size - 1)/ 2., (size-1)/2., size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx)+ np.square(yy)) / np.square(sigma))
    kernel /= np.sum(kernel)
    # Change kernel to PyTorch. reshapes to (channels, 1, size, size)
    kernel_tensor = torch.as_tensor(kernel, dtype=dtype)
    kernel_tensor = kernel_tensor.repeat(channels, 1 , 1, 1)
    kernel_tensor.to(device)
    return kernel_tensor

def gaussian_conv2d(x, g_kernel,stride, dtype=torch.float):
    #Assumes input of x is of shape: (minibatch, depth, height, width)
    #Infer depth automatically based on the shape
    groups = g_kernel.shape[0]
    channels = x.shape[1]
    x = x.repeat(1,groups//channels,1,1)
#     g_kernel = g_kernel.repeat_interleave(channels,dim=1)
    padding = g_kernel.shape[-1] // 2 # Kernel size needs to be odd number
    if len(x.shape) != 4:
        raise IndexError('Expected input tensor to be of shape: (batch, depth, height, width) but got: ' + str(x.shape))
#     print(groups)
    y = F.conv2d(x, weight=g_kernel, stride=stride,padding=padding, groups=groups)
    return y

def sample(img,kernel,mask):
#     print(img.shape)
    smoothed = gaussian_conv2d(img,kernel,1)
#     grid_size = smoothed.shape[-1]
    results = []
    for i in range(0,len(mask)):
        result_part = smoothed[:,i*3:(i+1)*3,:,:][:,:,mask[i]]
        results.append(result_part)
    results = torch.cat(results,dim=-1)
    return results

def generate_cor(img_size,max_ks = 7,r_step=31,phi_n = 78):
    center = (img_size//2,img_size//2)
    ra_1_range = np.linspace(0,1,r_step)
    ratio_range = ra_1_range**4
    cor_ls = []
    cor_ks_map= {}
    for i in range(0,len(ratio_range)):
        ratio = ratio_range[i]
        ra_1  = ra_1_range[i]
        r = ratio*(img_size//2)
        phi_range = np.linspace(0,1,int(phi_n*ra_1))[:-1]
        phi_range = phi_range*2*np.pi
        kernel_radius = ra_1*max_ks
        for phi in phi_range:
            x,y = pol2cart(r,phi)
            x_off = center[0]+int(x)
            y_off = center[1]+int(y)
            if x_off< img_size and y_off< img_size and (x_off,y_off) not in cor_ks_map:
                tip = (x_off,y_off,kernel_radius)
                cor_ls.append(tip)
                cor_ks_map[(x_off,y_off)] = kernel_radius
    return cor_ls,cor_ks_map


def generate_mask(cor_ls,img_size):
    cor_ls_split = {}
    for cor in cor_ls:
        kernel_size = max(int(cor[2]),1)
        if kernel_size not in cor_ls_split:
            cor_ls_split[kernel_size] = [cor]
        else:
            cor_ls_split[kernel_size].append(cor)
    mask_split = {}
    for kernel_size in cor_ls_split:
        cor_ls_ = cor_ls_split[kernel_size]
        sample_mask = torch.zeros((img_size,img_size))
        for cor in cor_ls_:
            x,y,_ = cor
            sample_mask[x,y]=1
        sample_mask = sample_mask==1
    #     kernel = gaussian_kernel(size=kernel_size,sigma=0.5)
        mask_split[kernel_size] = sample_mask
    mask = torch.stack(list(mask_split.values()))
#     keys = list(cor_ls_split.keys())
    return cor_ls_split,mask

def kernel_pyramid_gen(cor_ls_split):
    max_kernel_size = max(list(cor_ls_split.keys()))
    if max_kernel_size%2==0:
        max_kernel_size+=1
    kernel_ls = []
    for kernel_size in cor_ls_split:
        kernel = gaussian_kernel(size=kernel_size,sigma=0.5)
        left = (max_kernel_size-kernel.shape[-1])//2
        right = math.ceil((max_kernel_size-kernel.shape[-1])/2)
    #     print(left,right,(max_kernel_size-kernel.shape[-1])/2)
        kernel_ls.append(F.pad(kernel,(left,right,left,right)))
    #     break
    kernel_pyramid= torch.cat(kernel_ls,dim=0)
    return kernel_pyramid


# def corr_translate_gen(img_size,cor_ls,mask):
#     idx = torch.arange(img_size**2).reshape(img_size,img_size).to(torch.cfloat)
#     reweight = torch.tensor([cor[2] for cor in cor_ls])
    
# #     base = torch.tensor(0).to(torch.cfloat)
# #     base.imag +=1
#     idx_repeat = torch.stack([base*i+idx for i in keys])
#     corr_trans = idx_repeat[mask]
#     return corr_trans

def corr_translate_gen(img_size,mask):
#     max_kernel_size = int(max([cor[2] for cor in cor_ls]))
    idx = torch.arange(img_size**2).reshape(img_size,img_size).repeat(mask.shape[0],1,1)
    corr_trans = idx[mask]
    return corr_trans

def recon(img_size,pixel_va,corr_trans,cor_ks_map,min_pixel_size = 1):
    canvas = torch.zeros((3,img_size,img_size))
    count = torch.ones((3,img_size,img_size))*0.001
    for i in range(len(pixel_va)):
        pixel = pixel_va[i]
        xy = corr_trans[i]
#         xy = int(xy_k.real)
#         print(xy_k.imag)
        x_off = int(xy//img_size)
        y_off = int(xy%img_size)
        
        kernel_radius_int = max(int(cor_ks_map[(x_off,y_off)]),min_pixel_size)
#         print(kernel_radius_int//2,math.ceil(kernel_radius_int/2))
        canvas[:,x_off-kernel_radius_int//2:x_off+math.ceil(kernel_radius_int/2),y_off-kernel_radius_int//2:y_off+math.ceil(kernel_radius_int/2)] += pixel.reshape(-1,1,1)
        count[:,x_off-kernel_radius_int//2:x_off+math.ceil(kernel_radius_int/2),y_off-kernel_radius_int//2:y_off+math.ceil(kernel_radius_int/2)] +=1
    canvas = canvas/count
    toshow = canvas
    return canvas


img_size = (32+0*2)*3
upsize_ratio = 3
max_kernel_size=7
kernel_size = upsize_ratio
cor_ls,cor_ks_map = generate_cor(img_size,max_ks=max_kernel_size,r_step=38,phi_n = 80)
cor_ls_split,mask = generate_mask(cor_ls,img_size)
# mask= mask.to(torch.long)
keys = list(cor_ls_split.keys())
kernel_pyramid = kernel_pyramid_gen(cor_ls_split)
corr_trans = corr_translate_gen(img_size,mask)


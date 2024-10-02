import os
import shutil
import math
import torch
import argparse
import random
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import gc
from einops import rearrange
import wandb
from utils.spectral import get_weighted_cluster
# from utils.eval import WeightedKNNClassifier
from utils.eval import eval_model_KNN,eval_model_KNN_
from utils.others import *
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--max_device_batch_size', type=int, default=2000)
    parser.add_argument('--emb_dim', type=int, default=192)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--patch_size', type=int, default=2)
    parser.add_argument('--share', dest='share', action='store_true')
    parser.add_argument('--no-share', dest='share', action='store_false')
    parser.add_argument('--padding', type=int, default=1)
    parser.add_argument('--decoder_layer', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='vit-t-mae')
    parser.add_argument('--label_file', type=str, default='tmp/tmp_param/label_perm_true.npy')
    parser.add_argument('--epsilon', type=float, default=0.02)
    
    parser.add_argument('--n_clusters', type=int, default=64)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--reweight', type=int, default=0)
    parser.add_argument('--oracle', type=float, default=0.5)
    parser.add_argument('--w_factor', type=float, default=0.5)
    
    parser.add_argument('--num_groups', type=int, default=None)
#     parser.add_argument('--permute', type=int, default=None)


    parser.add_argument('--scale_lo', type=float, default=0.3)
    parser.add_argument('--scale_high', type=float, default=1.0)
    parser.add_argument('--ratio_lo', type=float, default=1.0)
    parser.add_argument('--ratio_high', type=float, default=1.0)
    parser.add_argument('--data_crop_rate', type=float, default=0.85)
    parser.add_argument('--flip', type=float, default=0.)
    
    parser.add_argument('--retina', dest='retina', action='store_true')
    parser.add_argument('--no-retina', dest='retina', action='store_false')
    

    parser.set_defaults(feature=True)
    args = parser.parse_args()
    set_seed(args.seed)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_path = "mae_batch_size_{}_decoder_layer_{}_mask_ratio_{}_reweight_{}_K_{}_oracle_{}_n_clusters_{}_seed_{}_w_factor_{}.pt".format(args.batch_size,args.decoder_layer,args.mask_ratio,args.reweight,args.K,args.oracle,args.n_clusters,args.seed,args.w_factor)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size
    
    if args.retina:
        print("retina")
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=
                                                     Compose([ToTensor(), 
                                                              RandomResizedCrop(32, scale=(args.scale_lo, args.scale_high), ratio=(args.ratio_lo, args.ratio_high)), 
                                                              transforms.RandomHorizontalFlip(p=args.flip),
                                                              Normalize(0.5, 0.5),
    #                                                           transforms.Pad(2,fill=0,padding_mode='edge'),
                                                              Upsample_transform(upsize_ratio)]))
        dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=8,pin_memory=True,drop_last=True)
        data_train,data_test,train_label,test_label = load_cifar_in_memory(kernel_pyramid,device =device)
    else:
        print("no retina")
        train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=
                                                     Compose([ToTensor(), 
                                                              RandomResizedCrop(32, scale=(args.scale_lo, args.scale_high), ratio=(args.ratio_lo, args.ratio_high)), 
                                                              transforms.RandomHorizontalFlip(p=args.flip),
                                                              Normalize(0.5, 0.5)]))
        dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4,drop_last=True)
        train_dataset_eval = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([ToTensor(), RandomResizedCrop(32, scale=(args.data_crop_rate, args.data_crop_rate), ratio=(1.0, 1.0)),transforms.RandomHorizontalFlip(p=args.flip), Normalize(0.5, 0.5)]))
        train_dl = torch.utils.data.DataLoader(train_dataset_eval, batch_size=load_batch_size, shuffle=False, num_workers=4)
        val_dataset_eval = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=Compose([ToTensor(), RandomResizedCrop(32, scale=(args.data_crop_rate, args.data_crop_rate), ratio=(1.0, 1.0)) ,Normalize(0.5, 0.5)]))
        train_label = torch.tensor(train_dataset_eval.targets)
        test_label = torch.tensor(val_dataset_eval.targets)
        test_dl = torch.utils.data.DataLoader(val_dataset_eval, batch_size=load_batch_size, shuffle=False, num_workers=4)
    
    
    
    if args.retina:
        from model.model_retina import MAE_ViT
        MI_score = np.load("tmp/tmp_param/MI_score_color_retina_fast_new_20000.npy")
        get_weighted_cluster(MI_score,args.K,args.oracle,args.w_factor,args.n_clusters,args.seed,args.label_file)
        model = MAE_ViT(args.label_file,mask_ratio=args.mask_ratio,
                        mlp_ratio=args.mlp_ratio,emb_dim=args.emb_dim,decoder_layer=args.decoder_layer,)    
       
        kernel_pyramid = kernel_pyramid.to(device)
    else:
        from model.model_magic import MAE_ViT
        idx = torch.from_numpy(np.load("tmp/tmp_param/32_perm_idx.npy"))
        gperm = GlobalPerm(idx)
        model = MAE_ViT(args.label_file,patch_size=args.patch_size,mask_ratio=args.mask_ratio,
                        mlp_ratio=args.mlp_ratio,emb_dim=args.emb_dim,decoder_layer=args.decoder_layer,
                        num_groups=args.num_groups)    
        model.forward_perm =gperm.forward_perm.clone()
        model.backward_perm =gperm.backward_perm.clone()
#     model = torch.nn.DataParallel(model).to(device)
    model = model.to(device)


    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)
    step_count = 0
    optim.zero_grad()
    wandb.init(project="mae_train_cifar10_station_",name = model_path[:-3])
    wandb.config.update(args)

    for e in range(args.total_epoch):
        print(e)
        model.train()
        losses = []
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            if args.retina:
                img = sample(img,kernel_pyramid,mask)
            else:
                img=gperm(img)
            predicted_img, mask_, features = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask_) / args.mask_ratio 
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        
        if  e%200==0:
            torch.save(model, "tmp/tmp_weight/"+model_path)
            if args.retina:
                eval_acc = eval_model_KNN(model,data_train,data_test,train_label,test_label,load_batch_size=load_batch_size,device = device)
            else:
                eval_acc = eval_model_KNN_(model,train_dl,test_dl,train_label,test_label,load_batch_size=load_batch_size,device = device,perm=gperm)
            wandb.log({'epoch':e, 'eval_acc': eval_acc, 'mae_loss': avg_loss, 'lr':lr_scheduler.get_last_lr()[0]})
            
        wandb.log({'epoch':e, 'mae_loss': avg_loss, 'lr':lr_scheduler.get_last_lr()[0]})
        ''' save model '''
#         torch.save(model.module, model_path)
        torch.save(model, "tmp/tmp_weight/"+model_path)


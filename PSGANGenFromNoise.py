#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:31:44 2019

@author: reiters
"""

from __future__ import print_function
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from utils import TextureDataset, setNoise, learnedWN
import torchvision.transforms as transforms
import torchvision.utils as vutils
import sys
from network import weights_init,Discriminator,calc_gradient_penalty,NetG
from config import opt,bMirror,nz,nDep,criterion
import time

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

canonicT=[transforms.RandomCrop(opt.imageSize),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
mirrorT= []
if bMirror:
    mirrorT += [transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip()]
transformTex=transforms.Compose(mirrorT+canonicT)
#dataset = TextureDataset(opt.texturePath,transformTex,opt.textureScale)
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
             #                            shuffle=True, num_workers=int(opt.workers))

N=0
ngf = int(opt.ngf)
ndf = int(opt.ndf)

desc="fc"+str(opt.fContent)+"_ngf"+str(ngf)+"_ndf"+str(ndf)+"_dep"+str(nDep)+"-"+str(opt.nDepD)

if opt.WGAN:
    desc +='_WGAN'
if opt.LS:
        desc += '_LS'
if bMirror:
    desc += '_mirror'
if opt.textureScale !=1:
    desc +="_scale"+str(opt.textureScale)
netD = Discriminator(ndf, opt.nDepD, bSigm=not opt.LS and not opt.WGAN)

##################################
netG =NetG(ngf, nDep, nz)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("device",device)

Gnets=[netG]
if opt.zPeriodic:
    Gnets += [learnedWN]

for net in [netD] + Gnets:
    try:
        net.apply(weights_init)
    except Exception as e:
        print (e,"weightinit")
    # passsetNoise
    pass
    net=net.to(device)
    print(net)

NZ = opt.imageSize//2**nDep
noise = torch.FloatTensor(opt.batchSize, nz, NZ,NZ)
fixnoise = torch.FloatTensor(1, nz, NZ*8,NZ*8)

real_label = 1
fake_label = 0

noise=noise.to(device)
fixnoise=fixnoise.to(device)

import numpy as np


#outModelName = '/home/reiters/psgan/famos/results/frankfurt_collection/2019-06-07_13-11-53/netG_epoch_99_fc1.0_ngf120_ndf120_dep5-5.pth'
#outModelName = '/gpfs/laur/sepia_tools/PSGAN_textures/usedTextures/netG_epoch_199_fc1.0_ngf80_ndf80_dep5-5.pth' 
outModelName =  '/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_rocks1_evaluated/netG_epoch_199_fc1.0_ngf80_ndf80_dep5-5.pth' 
#outModelName =  '/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_crack2_evaluated/best/netG_epoch_199_fc1.0_ngf80_ndf80_dep5-5.pth' 
#outModelName =  '/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_sand2_evaluated/best/netG_epoch_199_fc1.0_ngf80_ndf80_dep5-5.pth' 

netG.load_state_dict(torch.load(outModelName))
netG.eval()

epoch=500

#intNoise=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_sand2_evaluated/best/noiseImage1.npy',None, 'allow_pickle',True)
intNoise=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_rocks1_evaluated/noiseImage1.npy',None, 'allow_pickle',True)

for x in range(intNoise.shape[0]):
   # import pdb; pdb.set_trace()
    fixnoise=torch.from_numpy(intNoise[x]).float().to(device)
  #  fixnoise=setNoise(fixnoise)
   # np.save('%s/noiseBig_epoch_%03d_%s' % (opt.outputFolder, epoch, desc), fixnoise.cpu(),'allow_pickle',True)

    with torch.no_grad():
        fakeBig=netG(fixnoise)

    vutils.save_image(fakeBig,'%s/intTex_%03d_%s.jpg' % (opt.outputFolder, epoch,desc),normalize=True)
    epoch+=1

#netG.train()

#OPTIONAL
#save/load model for later use if desired
#outModelName = '%s/netG_epoch_%d_%s.pth' % (opt.outputFolder, epoch,desc)
#torch.save(netG.state_dict(),outModelName )
#netG.load_state_dict(torch.load(outModelName))

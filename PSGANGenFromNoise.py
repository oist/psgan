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
import numpy as np


ngf = 80
ndf = 80
nDep = 4
opt.nDepD = 4
opt.batchSize = 8
opt.zGL = 20
opt.zLoc = 10
opt.zPeriodic = 0
opt.sizeMultiplier=7
nz=opt.zGL+opt.zLoc+opt.zPeriodic

workPath = '/home/sam/bucket/textures/synthesizedImgs0-12/'


outModelName = workPath + 'usedImgs/netG_epoch_99.pth'
t1=np.load(workPath + 'usedImgs/noise1.npy',None,'allow_pickle',True)
t2=np.load(workPath + 'usedImgs/noise2.npy',None,'allow_pickle',True)


#linSpacing = np.linspace(-1,1,11)
#quadSpacing = np.square(linSpacing)
#t=np.append(0, np.cumsum(np.abs(np.diff(quadSpacing))))/2
#img1Ratio=t*0.35+.15

img1Ratio=np.linspace(0.15,0.45,10) # for synthesizedImgs0-12
#img1Ratio=np.linspace(0.25,.65,10) # for synthesizedImgs0-22


intNoise=[]
for x in range( len(img1Ratio)):
    intNoise.append(img1Ratio[x]*t1 + [1-img1Ratio[x]]*t2)
intNoise=np.array(intNoise)
np.save(workPath + 'usedImgs/intermediateNoise',intNoise)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True

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


netG.load_state_dict(torch.load(outModelName))
netG.eval()

epoch=0
for x in range(intNoise.shape[0]):

    fixnoise=torch.from_numpy(intNoise[x]).float().to(device)

    with torch.no_grad():
        fakeBig=netG(fixnoise)

    vutils.save_image(fakeBig,workPath + 'usedImgs/intTex_%03d.jpg' % (epoch),normalize=True)
    epoch+=1



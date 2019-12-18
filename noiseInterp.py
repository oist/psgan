#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:10:41 2019

@author: reiters
"""
import numpy as np


#t1=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/usedTextures/noiseBig_epoch_501_fc1.0_ngf80_ndf80_dep5-5.npy',None,'allow_pickle',True)
#t2=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/usedTextures/noiseBig_epoch_541_fc1.0_ngf80_ndf80_dep5-5.npy',None,'allow_pickle',True)
#t3=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/usedTextures/noiseBig_epoch_507_fc1.0_ngf80_ndf80_dep5-5.npy',None,'allow_pickle',True)
#t4=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/usedTextures/noiseBig_epoch_511_fc1.0_ngf80_ndf80_dep5-5.npy',None,'allow_pickle',True)
t1=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_rocks1_evaluated/noiseBig_epoch_500_fc1.0_ngf80_ndf80_dep5-5.npy',None,'allow_pickle',True)
t2=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_rocks1_evaluated/noiseBig_epoch_501_fc1.0_ngf80_ndf80_dep5-5.npy',None,'allow_pickle',True)
#t1=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_crack2_evaluated/best/noiseBig_epoch_512_fc1.0_ngf80_ndf80_dep5-5.npy',None,'allow_pickle',True)
#t2=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_crack2_evaluated/best/noiseBig_epoch_529_fc1.0_ngf80_ndf80_dep5-5.npy',None,'allow_pickle',True)
#t1=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_sand2_evaluated/best/noiseBig_epoch_500_fc1.0_ngf80_ndf80_dep5-5.npy',None,'allow_pickle',True)
#t2=np.load('/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_sand2_evaluated/best/noiseBig_epoch_529_fc1.0_ngf80_ndf80_dep5-5.npy',None,'allow_pickle',True)


img1Ratio=np.linspace(0,1,11) # for curtain-rocks
#img1Ratio=np.linspace(0.2,0.35,11) # for curtain-crack
#img1Ratio=np.linspace(0.4,.7,11) # for curtain-sand

intNoise=[]
for x in range( len(img1Ratio)):
    intNoise.append(img1Ratio[x]*t1 + [1-img1Ratio[x]]*t2)
    
#np.save('/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_crack2_evaluated/best/noiseImage1',intNoise)
#np.save('/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_sand2_evaluated/best/noiseImage1',intNoise)
np.save('/gpfs/laur/sepia_tools/PSGAN_textures/best_paired_models/curtain_rocks1_evaluated/noiseImage1',intNoise)
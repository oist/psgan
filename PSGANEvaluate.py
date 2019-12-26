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
import os


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

#outModelName = '/home/sam/bucket/textures/synthesizedImg0/modelTraining/netG_epoch_99_fc1.0_ngf80_ndf120_dep4-4.pth' 
#outModelName = '/home/sam/bucket/textures/synthesizedImg12/modelTraining/netG_epoch_99_fc1.0_ngf80_ndf80_dep4-4.pth' 
#outModelName = '/home/sam/bucket/textures/synthesizedImg20/modelTraining/netG_epoch_99_fc1.0_ngf80_ndf80_dep5-5.pth' 
#outModelName = '/home/sam/bucket/textures/synthesizedImg22/modelTraining/netG_epoch_99_fc1.0_ngf80_ndf80_dep4-4.pth' 
#outModelName = '/home/sam/bucket/textures/synthesizedImgs0-12/modelTraining/netG_epoch_99_fc1.0_ngf80_ndf80_dep4-4.pth' 
outModelName = '/home/sam/bucket/textures/synthesizedImgs0-22/modelTraining/netG_epoch_99_fc1.0_ngf80_ndf80_dep4-4.pth' 


opt.outputFolder = '/home/sam/bucket/textures/synthesizedImgs0-22/modelEvaluating/'

try:
    os.makedirs(opt.outputFolder)
except OSError:
    pass

text_file = open(opt.outputFolder+"options.txt", "w")
text_file.write(str(opt))
text_file.close()
print (opt)


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
cudnn.benchmark = True


N=0

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


sizeMultiplier=opt.sizeMultiplier#7 for curtain, 

fixnoise = torch.FloatTensor(1, nz, NZ*sizeMultiplier,NZ*sizeMultiplier)

real_label = 1
fake_label = 0

noise=noise.to(device)
fixnoise=fixnoise.to(device)

netG.load_state_dict(torch.load(outModelName))
netG.eval()

epoch=0
import numpy as np
for x in range(100):
   
    fixnoise=setNoise(fixnoise)
    np.save('%s/noiseBig_epoch_%03d_%s' % (opt.outputFolder, epoch, desc), fixnoise.cpu(),'allow_pickle',True)
        
    with torch.no_grad():
        fakeBig=netG(fixnoise)
    vutils.save_image(fakeBig,'%s/verybig_texture_%03d_%s.jpg' % (opt.outputFolder, epoch,desc),normalize=True)
    epoch+=1



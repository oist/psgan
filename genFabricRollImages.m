cd '/gpfs/laur/sepia_tools/textureDatabase/fabricRollImgs'

imgPix=1200

%rng(42)% roll 1
%rng(13) %roll 2
rng(7)% roll 3
imgList=randperm(30)-1;
imgList2=randperm(30)-1;


%% make list with neighbors above a certain distance in texture
texRep=h5read('/gpfs/laur/sepia_tools/textureDatabase/fabricRollImgs/vggTexRepresentations','/texRep');
texRep=double(texRep');
[mappedX, mapping] = pca(texRep, 50); %reduce things down in dimension
for x=1:(size(mappedX,1)/4)
    currRows=(4*(x-1)+1):4*x;
    meanTex(x,:)=mean(mappedX(currRows,:));
end

D=pdist2(meanTex, meanTex);
%relD=triu(D);
%relD(relD==0)=[];
%meanInterD=mean(relD);


list=randperm(30);
imgList=list(1);
Dtmp=D;

for i=2:30
    [~, maxD] = max(Dtmp(imgList(i-1),:))
    imgList(i)=maxD
    Dtmp(:,imgList(i-1))=0;
    Dtmp(imgList(i-1),:)=0;
end
imgList=imgList-1;

%second copy
list=randperm(30);
imgList2=list(1);
Dtmp=D;

for i=2:30
    [~, maxD] = max(Dtmp(imgList2(i-1),:))
    imgList2(i)=maxD
    Dtmp(:,imgList2(i-1))=0;
    Dtmp(imgList2(i-1),:)=0;
end
imgList2=imgList2-1;

%%



bigList=[imgList'; imgList2'];

%imgSize=zeros(numel(bigList),3)
bigImage=zeros(imgPix,imgPix*numel(bigList),3,'uint8');
for img=1:numel(bigList)
    currIm=imread(['imgCrop' num2str(bigList(img)) '.png']);
    % imgSize(img,:)=size(currIm)
    resizedimage = imresize(currIm, [imgPix imgPix]);
    bigImage(:,(imgPix*(img-1)+1):(imgPix*(img)),:)=resizedimage;
    img
end
bigImage2=[zeros(imgPix,imgPix*4,3,'uint8') bigImage 255*ones(imgPix,imgPix*4,3,'uint8')];
imwrite(bigImage2,'roll2x_3_distanceMaintained.png')







%  imshow(bigImage2)
%  print('roll2x_1.pdf','-dpdf','-r400','-bestfit')

%% For synthesized image roll
imgPix=1120


imgTensor= zeros(29,imgPix,imgPix,3,'uint8');

cd('/home/sam/bucket/textures/synthesizedImgs0-12/usedImgs')
for img = 0:9
    currIm=imread(['intTex_00' num2str(img) '.jpg']);
    resizedimage = imresize(currIm, [imgPix imgPix]);
    imgTensor(img+1,:,:,:)=resizedimage;
end

cd('/home/sam/bucket/textures/synthesizedImgs0-22/usedImgs')
for img = 0:9
    currIm=imread(['intTex_00' num2str(img) '.jpg']);
    resizedimage = imresize(currIm, [imgPix imgPix]);
    imgTensor(img+11,:,:,:)=resizedimage;
end

cd('/home/sam/bucket/textures/synthesizedImg0/usedImgs')
currIm=imread(['verybig_texture_046.jpg']);
resizedimage = imresize(currIm, [imgPix imgPix]);
imgTensor(21,:,:,:)=resizedimage;
currIm=imread(['verybig_texture_090.jpg']);
resizedimage = imresize(currIm, [imgPix imgPix]);
imgTensor(22,:,:,:)=resizedimage;
cd('/home/sam/bucket/textures/synthesizedImg0/originalImg')
currIm=imread(['imgCrop0.png']);
resizedimage = imresize(currIm, [imgPix imgPix]);
imgTensor(23,:,:,:)=resizedimage;

cd('/home/sam/bucket/textures/synthesizedImg12/usedImgs')
currIm=imread(['verybig_texture_010.jpg']);
resizedimage = imresize(currIm, [imgPix imgPix]);
imgTensor(24,:,:,:)=resizedimage;
currIm=imread(['verybig_texture_015.jpg']);
resizedimage = imresize(currIm, [imgPix imgPix]);
imgTensor(25,:,:,:)=resizedimage;
cd('/home/sam/bucket/textures/synthesizedImg12/originalImg')
currIm=imread(['imgCrop12.png']);
resizedimage = imresize(currIm, [imgPix imgPix]);
imgTensor(26,:,:,:)=resizedimage;

cd('/home/sam/bucket/textures/synthesizedImg22/usedImgs')
currIm=imread(['verybig_texture_013.jpg']);
resizedimage = imresize(currIm, [imgPix imgPix]);
imgTensor(27,:,:,:)=resizedimage;
currIm=imread(['verybig_texture_020.jpg']);
resizedimage = imresize(currIm, [imgPix imgPix]);
imgTensor(28,:,:,:)=resizedimage;
cd('/home/sam/bucket/textures/synthesizedImg22/originalImg')
currIm=imread(['imgCrop22.png']);
resizedimage = imresize(currIm, [imgPix imgPix]);
imgTensor(29,:,:,:)=resizedimage;




bigImage=zeros(imgPix,imgPix*29,3,'uint8');
rng(7)
imgList=randperm(29)
imgList=1:29
for img=1:size(imgTensor,1)
     bigImage(:,(imgPix*(img-1)+1):(imgPix*(img)),:)=squeeze(imgTensor(imgList(img),:,:,:));
end

bigImage2=[zeros(imgPix,imgPix*4,3,'uint8') bigImage 255*ones(imgPix,imgPix*4,3,'uint8')];
imwrite(bigImage2,'rollSynthesized_nonrandom.png')


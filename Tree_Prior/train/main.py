from skimage import morphology
import sys
import numpy
import torch
import torch.nn as nn
import numpy as np
from UNet import UNet
import segUtil

def topologyloss(output,label):
    loss=0
    output,label=output.to('cpu'),label.to('cpu')
    output,label=output.data.numpy(),label.data.numpy()
    batchsize=output.shape[0]
    for i in range(batchsize):
        VP,VL=output[i,0,:,:,:],label[i,0,:,:,:]
        vp=np.where(VP>0.5,1,0)
        sp = morphology.skeletonize(vp).astype(np.uint8)
        vp[sp==0]=0
        SP=vp
        SL = morphology.skeletonize(VL).astype(np.uint8)
        SL=np.where(SL==0,0,1)
        Tprec=np.sum(SP*VL)/(np.sum(SP)+torch.finfo(torch.float32).eps)
        Tsens=np.sum(SL*VP)/(np.sum(SL)+torch.finfo(torch.float32).eps)
        loss+=1-2*Tprec*Tsens/((Tprec+Tsens)+torch.finfo(torch.float32).eps)
    return loss/batchsize

# use GPU?
GPUID = 0
device = torch.device("cuda:"+str(GPUID) if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")

# meta parameters
patchSize = 64
numEPOCH = 80000000
BATCH_SIZE = 16
LR = 1e-4
includeNegativePatchRatio = 0.1
# instantiate network
#
lrs = []

# train
if __name__ == "__main__":

    from Getdataname import allTrainNames, allTrainNamesMask
    # allTrainNames=['/data/zhuhaoran/segmentionPA/dataset/test/PA000005/image/PA000005.nii.gz']
    # allTrainNamesMask=['/data/zhuhaoran/segmentionPA/dataset/test/PA000005/label/PA000005.nii.gz']
    allImages, allMasks = segUtil.loadAllImagesAndMasks(allTrainNames, allTrainNamesMask, patchSize)
    
    n = len(allImages)
    print(n)
    if len(sys.argv) <= 1:
        net = UNet(in_channels=1,n_classes=1).to(device)
    else:
        modelName = sys.argv[1]
        net = torch.load(modelName, map_location = device)
        net.train()
    bceloss = nn.BCELoss() 
    clwight=0
    optimizer = torch.optim.Adam(net.parameters(), lr = LR)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)



    bestLoss = 1e6
    for iEpoch in range(numEPOCH):
        #print("iEpoch", iEpoch)
        #pIdx = numpy.random.permutation(n)[:BATCH_SIZE]
        #print(type(p1), p1.shape)
        imgIdx = numpy.random.randint(0, n, size = 1)
        imgIdx = imgIdx[0]

        trainImageBatch, trainMask = segUtil.getPatchBatchFromMemory(imgIdx, allImages, allMasks, patchSize, batch_size = BATCH_SIZE, negativePatchRatio = includeNegativePatchRatio)

       

        # threshold=numpy.where(((trainImageBatch>200)&(trainImageBatch<500)),1,0)
        trainImageBatch = torch.from_numpy(trainImageBatch)
        trainMask = torch.from_numpy(trainMask)
        # threshold = torch.from_numpy(threshold).float()
        # trainImageBatch=trainImageBatch.to(device)
        # inputs=torch.cat([trainImageBatch,threshold.to(device)],dim=1)
        inputs, labels = trainImageBatch.to(device), trainMask.to(device)
        for i in range(1):
            optimizer.zero_grad()
            outputs = net(inputs)
            #print(outputs.max())
            # inputs=torch.cat([trainImageBatch,outputs.data],dim=1)
        #print(torch.max(outputs).item())
        #print(torch.min(outputs).item())
        #print(type(inputs), inputs.type(), labels.type(), labels.size(), outputs.size(), outputs.type())
            loss= topologyloss(outputs,labels)
            loss.backward()
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

            if loss.item() < bestLoss:
                bestLoss = loss.item()
                torch.save(net, "/Unet-cldice.pth")
            print("epoch", iEpoch,  "loss = ", loss.item())

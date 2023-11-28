################################################################################
# This is training of Unet with cosine scheduler.
#
# Also this saves the best-so-far model
#
# This is the one I'm using the most
################################################################################
from skimage import morphology
import sys
import numpy
#import SimpleITK as sitk

import torch
import torch.nn as nn
#import torch.optim as optim
import numpy as np
from scipy.stats import norm
from scipy.ndimage import distance_transform_edt
from UNet import UNet
import segUtil


def twloss(outputs, labels, weights):
    # Flatten the outputs and labels to apply the element-wise operations.
    BATCH_SIZE = outputs.shape[0]
    outputs_flat = outputs.view(-1)
    labels_flat = labels.view(-1)
    weights_flat = weights.view(-1)

    # Calculate the weighted intersection and the weighted union.
    weighted_intersection = torch.sum(weights_flat * outputs_flat * labels_flat)
    weighted_union = torch.sum(weights_flat * (outputs_flat + labels_flat))

    # Calculate the Topology-Aware Loss using the provided formula.
    loss = 1 - (2 * weighted_intersection / weighted_union)
    
    return loss/BATCH_SIZE
# use GPU?
GPUID = 2
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
    allImages, allMasks ,allWeights = segUtil.loadAllImagesAndMasks_weight(allTrainNames, allTrainNamesMask, patchSize,sigma,p)
    n = len(allImages)
    print(n)
    if len(sys.argv) <= 1:
        net = UNet(in_channels=1,n_classes=1).to(device)
    else:
        modelName = sys.argv[1]
        net = torch.load(modelName, map_location = device)
        net.train()
    #net = torch.load('/data/zhuhaoran/segmentionPA/modelpth/w1p10nolabel.pth', map_location=torch.device(device))
      # MSE for segmetnation
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

        trainImageBatch, trainMask ,trainWeight = segUtil.getPatchBatchFromMemory_weight(imgIdx, allImages, allMasks,allWeights, patchSize, batch_size = BATCH_SIZE, negativePatchRatio = includeNegativePatchRatio)

       

        # threshold=numpy.where(((trainImageBatch>200)&(trainImageBatch<500)),1,0)
        trainImageBatch = torch.from_numpy(trainImageBatch)
        trainMask = torch.from_numpy(trainMask)
        trainWeight=torch.from_numpy(trainWeight)
        # threshold = torch.from_numpy(threshold).float()
        # trainImageBatch=trainImageBatch.to(device)
        # inputs=torch.cat([trainImageBatch,threshold.to(device)],dim=1)
        inputs, labels = trainImageBatch.to(device), trainMask.to(device)
        trainWeight=trainWeight.to(device)
        for i in range(1):
            optimizer.zero_grad()
            outputs = net(inputs)
            #print(outputs.max())
            # inputs=torch.cat([trainImageBatch,outputs.data],dim=1)
        #print(torch.max(outputs).item())
        #print(torch.min(outputs).item())
        #print(type(inputs), inputs.type(), labels.type(), labels.size(), outputs.size(), outputs.type())
            #print(trainWeight.max(),trainWeight.min())
            loss = twloss(outputs, labels, trainWeight)

            loss.backward()
            optimizer.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            #print("Factor = ", 0.65 ** iEpoch," , Learning Rate = ", optimizer.param_groups[0]["lr"])
            scheduler.step()
            #print('[%d, %d] loss: %.03f' % (iEpoch + 1, sum_loss / 100))
            if loss.item() < bestLoss:
                bestLoss = loss.item()
                torch.save(net, "model.pth")
            print("epoch", iEpoch, "loss = ", loss.item())


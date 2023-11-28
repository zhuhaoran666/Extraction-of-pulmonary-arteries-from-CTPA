from pyexpat import model
import sys
import SimpleITK as sitk

import numpy

import math

import numpy.random
from torch import nn
import torch


def segment3DPatchBatch(model, inputImagePatchBatchArray, GPUid = 0):

    device = torch.device("cuda:" + str(GPUid) if torch.cuda.is_available() else "cpu")



    threshold=numpy.where(((inputImagePatchBatchArray>200)&(inputImagePatchBatchArray<500)),1,0)

    threshold = torch.from_numpy(threshold).float()

    inputImagePatchBatchArray = torch.from_numpy(inputImagePatchBatchArray)
    inputs=inputImagePatchBatchArray.to(device)
    inputs = inputs.to(device)
    with torch.no_grad():
        results = model(inputs)

    outputSegBatchArray = results.cpu().numpy()

    outputSegBatchArray = outputSegBatchArray[:, 0, :, :, :]


    return outputSegBatchArray



def segment3DImageRandomSampleDividePrior(model, imageArray, patchSideLen = 64, numPatchSampleFactor = 10,
                                             batch_size = 1, num_segmentation_classes = 1, GPUid = 0):
    sz = imageArray.shape
    print("imageArray.shape = ", sz)
    numChannel = 1 # for gray
    iChannel = 0 # for gray
    numPatchSample = math.ceil((sz[0]/patchSideLen)*(sz[1]/patchSideLen)*(sz[2]/patchSideLen)*numPatchSampleFactor)


    segArray = numpy.zeros((num_segmentation_classes, sz[0], sz[1], sz[2]), dtype=numpy.float32)
    priorImage = numpy.zeros((sz[0], sz[1], sz[2]), dtype=numpy.float32)

    patchShape = (patchSideLen, patchSideLen, patchSideLen)
    imagePatchBatch = numpy.zeros((batch_size, numChannel, patchShape[0], patchShape[1], patchShape[2]), dtype=numpy.float32)

    for itPatch in range(0, numPatchSample, batch_size):

        allPatchTopLeftX = numpy.random.randint(0, sz[0] - patchShape[0], size = batch_size)
        allPatchTopLeftY = numpy.random.randint(0, sz[1] - patchShape[1], size = batch_size)
        allPatchTopLeftZ = numpy.random.randint(0, sz[2] - patchShape[2], size = batch_size)

        for itBatch in range(batch_size):
            thisTopLeftX = allPatchTopLeftX[itBatch]
            thisTopLeftY = allPatchTopLeftY[itBatch]
            thisTopLeftZ = allPatchTopLeftZ[itBatch]

            #: ad hoc: 0 in channel axis coz only gray image
            tmp = imageArray[thisTopLeftX:(thisTopLeftX + patchShape[0]),
                             thisTopLeftY:(thisTopLeftY + patchShape[1]),
                             thisTopLeftZ:(thisTopLeftZ + patchShape[2])]

            imagePatchBatch[itBatch, iChannel, :, :, :] = tmp

        #imagePatchBatch = imagePatchBatch
        segBatch = segment3DPatchBatch(model, imagePatchBatch, GPUid = GPUid)

        for itBatch in range(batch_size):
            thisTopLeftX = allPatchTopLeftX[itBatch]
            thisTopLeftY = allPatchTopLeftY[itBatch]
            thisTopLeftZ = allPatchTopLeftZ[itBatch]

            segArray[:, thisTopLeftX:(thisTopLeftX + patchShape[0]),
                     thisTopLeftY:(thisTopLeftY + patchShape[1]),
                     thisTopLeftZ:(thisTopLeftZ + patchShape[2])] += segBatch[itBatch, :, :, :]
            priorImage[thisTopLeftX:(thisTopLeftX + patchShape[0]),
                       thisTopLeftY:(thisTopLeftY + patchShape[1]),
                       thisTopLeftZ:(thisTopLeftZ + patchShape[2])] += numpy.ones((patchShape[0], patchShape[1], patchShape[2]))

    for it in range(num_segmentation_classes):
        segArray[it, :, :, :] /= (priorImage + numpy.finfo(numpy.float32).eps)
        segArray[it, :, :, :]*=100

    print(numpy.max(segArray[0,:, :, :]),numpy.min(segArray[0,:, :, :]))
    # segArray=numpy.where(segArray>50,1,0)
    outputSegArrayOfObject1 = segArray[0, :, :, :]
    return outputSegArrayOfObject1
import segUtil
def segmentCurveRes4(modelName, inputImageName, outputImageName, GPUid = 0, patchSize = 64):
    net = torch.load(modelName, map_location=torch.device('cuda:' + str(GPUid)))
    net.eval()
    testIm = sitk.ReadImage(inputImageName, sitk.sitkFloat64)
    testIm = segUtil.mirrorPad(testIm, patchSize) 
    testImg = sitk.GetArrayFromImage(testIm)
    testImg = segUtil.adjustGrayImage(testImg)
    testImgSeg = segment3DImageRandomSampleDividePrior(model = net, imageArray = testImg, patchSideLen = patchSize, numPatchSampleFactor = 30, batch_size = 1, num_segmentation_classes = 1, GPUid = GPUid)
    outputSeg = sitk.GetImageFromArray(testImgSeg.astype(numpy.uint8), isVector=False)
    outputSeg.SetSpacing(testIm.GetSpacing())
    outputSeg.SetOrigin(testIm.GetOrigin())
    outputSeg.SetDirection(testIm.GetDirection())
    sitk.WriteImage(outputSeg, outputImageName, True)
if __name__ == "__main__":

    GPUID = 2
    patchSize = 64
    modelName='model.pth'
    testImageName ="PA.nrrd"
    outputName = "PA.seg.nrrd"
    segmentCurveRes4(modelName, testImageName, outputName, GPUid = GPUID, patchSize = patchSize)


import numpy
import SimpleITK as sitk
def adjustGrayImage(img):
    tmp = img.astype(numpy.float32)
    p10 = numpy.percentile(tmp, 10)
    p90 = numpy.percentile(tmp, 90)
    img = (tmp - p10)/(p90 - p10)
    return img
def adjustLabelImage(mask):
    mask = mask.astype(numpy.float32)
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask
def adjustData(img, mask):
    img = adjustGrayImage(img)
    mask = adjustLabelImage(mask)
    return (img,mask)
def getPatchBatchFromFile(thisImageName, thisMaskName, patchSize, batch_size):
    thisImg = sitk.ReadImage(thisImageName, sitk.sitkFloat64)
    thisMask = sitk.ReadImage(thisMaskName, sitk.sitkFloat64)
    thisImg = sitk.GetArrayFromImage(thisImg)
    thisMask = sitk.GetArrayFromImage(thisMask)
    numChannels = 1 
    imageBatch = numpy.zeros((batch_size, numChannels, patchSize, patchSize, patchSize), dtype=numpy.float32)
    maskBatch = numpy.zeros((batch_size, numChannels, patchSize, patchSize, patchSize), dtype=numpy.float32)
    n = thisImg.shape[0]
    randIdxForThisBatch = numpy.random.randint(0, n-1, size = 1)
    randomIdx = randIdxForThisBatch[0]
    sz = thisImg.shape
    numValidPatch = 0
    while numValidPatch < batch_size:
        allPatchTopLeftX = numpy.random.randint(0, sz[0] - patchSize, size = 1)
        allPatchTopLeftY = numpy.random.randint(0, sz[1] - patchSize, size = 1)
        allPatchTopLeftZ = numpy.random.randint(0, sz[2] - patchSize, size = 1)
        thisTopLeftX = allPatchTopLeftX[0]
        thisTopLeftY = allPatchTopLeftY[0]
        thisTopLeftZ = allPatchTopLeftZ[0]
        thisLabelImPatch = thisMask[thisTopLeftX:(thisTopLeftX + patchSize), thisTopLeftY:(thisTopLeftY + patchSize), thisTopLeftZ:(thisTopLeftZ + patchSize)]
        if thisLabelImPatch.max() > 0:
            thisImPatch = thisImg[thisTopLeftX:(thisTopLeftX + patchSize), thisTopLeftY:(thisTopLeftY + patchSize), thisTopLeftZ:(thisTopLeftZ + patchSize)]
            imageBatch[numValidPatch, 0, :, :, :] = thisImPatch.astype(numpy.float32)
            maskBatch[numValidPatch, 0, :, :, :] = thisLabelImPatch.astype(numpy.float32)
            numValidPatch += 1
    return imageBatch,maskBatch
def mirrorPad(img, patchSize):
    sz = img.GetSize()
    padUpperBound = [0, 0, 0]
    for i in range(3):
        if sz[i] < patchSize:
            padUpperBound[i] = 10 + patchSize - sz[i]
    mirrorPad = sitk.MirrorPadImageFilter()
    mirrorPad.SetPadUpperBound(padUpperBound)
    newImg = mirrorPad.Execute(img)
    return newImg
def loadAllImagesAndMasks(allImageNames, allMaskImageNames, patchSize):
    n = len(allImageNames)
    print("num of images = ", n, "len(allTrainNamesMask)", len(allMaskImageNames))
    imageAll = []
    maskAll = []
    for it in range(n):
        thisMaskName = allMaskImageNames[it]
        thisImageName = allImageNames[it]
        thisImg = sitk.ReadImage(thisImageName, sitk.sitkFloat64)
        thisMask = sitk.ReadImage(thisMaskName, sitk.sitkFloat64)
        thisImg = mirrorPad(thisImg, patchSize)
        thisMask = mirrorPad(thisMask, patchSize)
        thisImg = sitk.GetArrayFromImage(thisImg)
        thisMask = sitk.GetArrayFromImage(thisMask)
        thisImg, thisMask = adjustData(thisImg, thisMask)
        imageAll.append(thisImg)
        maskAll.append(thisMask)
    return imageAll, maskAll
def getPatchBatchFromMemory(it, imageBatch, maskBatch, patchSize, batch_size, negativePatchRatio = 0):
    thisImg = imageBatch[it]
    thisMask = maskBatch[it]
    numChannels = 1 
    imageBatch = numpy.zeros((batch_size, numChannels, patchSize, patchSize, patchSize), dtype=numpy.float32)
    maskBatch = numpy.zeros((batch_size, numChannels, patchSize, patchSize, patchSize), dtype=numpy.float32)
    sz = thisImg.shape

    numValidPatch = 0
    while numValidPatch < batch_size:
        allPatchTopLeftX = numpy.random.randint(0, sz[0] - patchSize, size = 1)
        allPatchTopLeftY = numpy.random.randint(0, sz[1] - patchSize, size = 1)
        allPatchTopLeftZ = numpy.random.randint(0, sz[2] - patchSize, size = 1)
        thisTopLeftX = allPatchTopLeftX[0]
        thisTopLeftY = allPatchTopLeftY[0]
        thisTopLeftZ = allPatchTopLeftZ[0]
        thisLabelImPatch = thisMask[thisTopLeftX:(thisTopLeftX + patchSize), thisTopLeftY:(thisTopLeftY + patchSize), thisTopLeftZ:(thisTopLeftZ + patchSize)]
        probIncludeNegativePatch = numpy.random.rand(1)
        if thisLabelImPatch.max() > 0 or probIncludeNegativePatch[0] < negativePatchRatio:
            thisImPatch = thisImg[thisTopLeftX:(thisTopLeftX + patchSize), thisTopLeftY:(thisTopLeftY + patchSize), thisTopLeftZ:(thisTopLeftZ + patchSize)]
            imageBatch[numValidPatch, 0, :, :, :] = thisImPatch.astype(numpy.float32)
            maskBatch[numValidPatch, 0, :, :, :] = thisLabelImPatch.astype(numpy.float32)
            numValidPatch += 1
    return imageBatch,maskBatch
from skimage import morphology
import numpy as np
from scipy.stats import norm
from scipy.ndimage import distance_transform_edt
def loadAllImagesAndMasks_weight(allImageNames, allMaskImageNames, patchSize,sigma,p):
    n = len(allImageNames)
    print("num of images = ", n, "len(allTrainNamesMask)", len(allMaskImageNames))
    imageAll = []
    maskAll = []
    weightAll = []
    for it in range(n):
        thisMaskName = allMaskImageNames[it]
        thisImageName = allImageNames[it]
        thisImg = sitk.ReadImage(thisImageName, sitk.sitkFloat64)
        thisMask = sitk.ReadImage(thisMaskName, sitk.sitkFloat64)
        thisImg = mirrorPad(thisImg, patchSize)
        thisMask = mirrorPad(thisMask, patchSize)
        thisImg = sitk.GetArrayFromImage(thisImg)
        thisMask = sitk.GetArrayFromImage(thisMask)
        thisImg, thisMask = adjustData(thisImg, thisMask)
        thisweight=count_weight1(thisMask,sigma,p)
        imageAll.append(thisImg)
        maskAll.append(thisMask)
        weightAll.append(thisweight)
        
    return imageAll, maskAll, weightAll



def getPatchBatchFromMemory_weight(it, imageBatch, maskBatch,weightBatch, patchSize, batch_size, negativePatchRatio = 0.1):
    thisImg = imageBatch[it]
    thisMask = maskBatch[it]
    thisWeight= weightBatch[it]
    numChannels = 1 # for gray image
    imageBatch = numpy.zeros((batch_size, numChannels, patchSize, patchSize, patchSize), dtype=numpy.float32)
    maskBatch = numpy.zeros((batch_size, numChannels, patchSize, patchSize, patchSize), dtype=numpy.float32)
    weightBatch= numpy.zeros((batch_size, numChannels, patchSize, patchSize, patchSize), dtype=numpy.float32)
    sz = thisImg.shape
    numValidPatch = 0
    while numValidPatch < batch_size:
        allPatchTopLeftX = numpy.random.randint(0, sz[0] - patchSize, size = 1)
        allPatchTopLeftY = numpy.random.randint(0, sz[1] - patchSize, size = 1)
        allPatchTopLeftZ = numpy.random.randint(0, sz[2] - patchSize, size = 1)

        thisTopLeftX = allPatchTopLeftX[0]
        thisTopLeftY = allPatchTopLeftY[0]
        thisTopLeftZ = allPatchTopLeftZ[0]

        thisLabelImPatch = thisMask[thisTopLeftX:(thisTopLeftX + patchSize), thisTopLeftY:(thisTopLeftY + patchSize), thisTopLeftZ:(thisTopLeftZ + patchSize)]
        thisWeightPatch = thisWeight[thisTopLeftX:(thisTopLeftX + patchSize), thisTopLeftY:(thisTopLeftY + patchSize), thisTopLeftZ:(thisTopLeftZ + patchSize)]
        probIncludeNegativePatch = numpy.random.rand(1)
        if thisLabelImPatch.max() > 0 or probIncludeNegativePatch[0] < negativePatchRatio:
            thisImPatch = thisImg[thisTopLeftX:(thisTopLeftX + patchSize), thisTopLeftY:(thisTopLeftY + patchSize), thisTopLeftZ:(thisTopLeftZ + patchSize)]
            imageBatch[numValidPatch, 0, :, :, :] = thisImPatch.astype(numpy.float32)
            maskBatch[numValidPatch, 0, :, :, :] = thisLabelImPatch.astype(numpy.float32)
            weightBatch[numValidPatch, 0, :, :, :] = thisWeightPatch.astype(numpy.float32)
            numValidPatch += 1
    return imageBatch,maskBatch,weightBatch
def count_values(arr):
    count_dict = {}
    for val in arr:
        if val in count_dict:
            count_dict[val] += 1
        else:
            count_dict[val] = 1
    return count_dict

def distance_weighted_array(label,centerline,sigma):
    d_min = distance_transform_edt(1-centerline)
    weights = np.exp(-sigma*d_min)
    return weights
def count_weight1(label,sigma=10):
    sk=morphology.skeletonize(label)
    sk=np.where(sk>0,1,0)
    weight=distance_weighted_array(label,sk,sigma)
    return weight

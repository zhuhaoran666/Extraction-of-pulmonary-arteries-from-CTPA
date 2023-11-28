import numpy as np
import SimpleITK as sitk
from sklearn.metrics import confusion_matrix
from scipy import ndimage
import scipy
from skimage import morphology
import numpy as np
from scipy.spatial.distance import directed_hausdorff
def evaluate(ExpertSegPath,AutoSegPath):
    ExpertSeg = sitk.ReadImage(ExpertSegPath, sitk.sitkInt8)
    AutoSeg = sitk.ReadImage(AutoSegPath, sitk.sitkInt8)
    Spacing=ExpertSeg.GetSpacing()
    ExpertSeg = sitk.GetArrayFromImage(ExpertSeg)
    AutoSeg = sitk.GetArrayFromImage(AutoSeg)
    structure = scipy.ndimage.generate_binary_structure(3, 3)
    # _,num_featuresex = scipy.ndimage.label(ExpertSeg,structure=structure)
    # _,num_featuresat = scipy.ndimage.label(AutoSeg,structure=structure)
    num_featuresat=0
    num_featuresex=0
    result_coordinates = np.argwhere(AutoSeg)
    ground_truth_coordinates = np.argwhere(ExpertSeg)
    distance = [directed_hausdorff(result_coordinates, ground_truth_coordinates)[0]]
    sorted_distances = np.sort(distance)
    distance = np.percentile(sorted_distances, 95)
    VP,VL=AutoSeg,ExpertSeg
    vp=np.where(VP>0.5,1,0)

    SP = morphology.skeletonize(VP).astype(np.uint8)
    SL = morphology.skeletonize(VL).astype(np.uint8)

    Tprec=np.sum(SP*VL)/(np.sum(SP))
    Tsens=np.sum(SL*VP)/(np.sum(SL))
    cldice=2*Tprec*Tsens/((Tprec+Tsens))    

    list=[]
    for i in range(0,num_featuresat):
        if np.count_nonzero(_==(i+1))>10:
            list.append(np.count_nonzero(_==(i+1)))
    ExpertSeg=ExpertSeg.flatten()
    AutoSeg=AutoSeg.flatten()
    confusion=confusion_matrix(ExpertSeg,AutoSeg)
    TP=confusion[1,1]
    TN=confusion[0,0]
    FP=confusion[0,1]
    FN=confusion[1,0]
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    IoU=TP/(TP+FP+FN)
    dice=2*TP/(2*TP+FP+FN)
    return confusion,precision,recall,IoU,dice,cldice,abs(num_featuresex-num_featuresat),distance
from tqdm import tqdm 
testlist=['05','16','24',26,27,36,38,42,46,47,53,56,60,63,70,73,74,78,80,82]
# testlist=['05','16']
precision_sum=np.zeros(len(testlist))
IoU_sum=np.zeros(len(testlist))
dice_sum=np.zeros(len(testlist))
recall_sum=np.zeros(len(testlist))
betti0_sum=np.zeros(len(testlist))
cldice_sum=np.zeros(len(testlist))
distance_sum=np.zeros(len(testlist))
k=0
for i in tqdm(testlist):
    ExpertSegPath="PA0000"+str(i)+"/label/PA0000"+str(i)+".nii.gz"
    AutoSegPath="PA0000"+str(i)+".seg.nrrd"
    _,precision,recall,IoU,dice,cldice,betti0,distance=evaluate(ExpertSegPath,AutoSegPath)
    precision_sum[k]=precision
    IoU_sum[k]=IoU
    dice_sum[k]=dice
    recall_sum[k]=recall 
    cldice_sum[k]=cldice
    betti0_sum[k]=betti0
    distance_sum[k]=distance
    k=k+1
print("precision mean: %.4f" % precision_sum.mean(),"std: %.4f" % precision_sum.std())
print("recall mean: %.4f" % recall_sum.mean(),"std: %.4f" % recall_sum.std())
print("IoU mean: %.4f" % IoU_sum.mean(),"std: %.4f" % IoU_sum.std())
print("dice mean: %.4f" % dice_sum.mean(),"std: %.4f" % dice_sum.std())
print("cldice mean: %.4f" % cldice_sum.mean(),"std: %.4f" % cldice_sum.std())
print("betti0 mean: %.4f" % betti0_sum.mean(),"std: %.4f" % betti0_sum.std())
print("distance mean: %.4f" % distance_sum.mean(),"std: %.4f" % distance_sum.std())
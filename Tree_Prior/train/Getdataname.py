import os
def GetImageName(NumList,Pth,type):
    Names=[]
    for i in NumList:
        if i<10:
            i='0'+str(i)
        pth=Pth+'PA0000'+str(i)+'/image/PA0000'+str(i)+type
        Names.append(pth)
    return Names

def GetLabelName(NumList,Pth,type):
    Names=[]
    for i in NumList:
        if i<10:
            i='0'+str(i)
        pth=Pth+'PA0000'+str(i)+'/label/PA0000'+str(i)+type
        Names.append(pth)
    return Names

trainlist=[5,16,24,26,27,36,38,42,46,47,53,56,60,63,70,73,74,78]
import os
import glob
allTrainNames=[]
allTrainNamesMask=[]
train_path='dataset/train'
train_name=os.listdir(train_path)
for name in train_name:
    name=name.split('.')[0]
    allTrainNames.append(os.path.join(train_path,name,'image',name+'.nii.gz'))
    allTrainNamesMask.append(os.path.join(train_path,name,'label',name+'.nii.gz'))
print("len(allTrainNames) = ", len(allTrainNames), "len(allTrainNamesMask) = ", len(allTrainNamesMask))




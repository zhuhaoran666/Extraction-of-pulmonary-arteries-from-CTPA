import SimpleITK as sitk
import cv2
import numpy as np
from matplotlib.pyplot import *
from skimage import morphology
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
import cv2
import numpy as np
from matplotlib.pyplot import *
from scipy.stats import norm
from scipy.ndimage import distance_transform_edt
import numpy as np
from sklearn.linear_model import LinearRegression
def connected_components3D(image):
    connected6=[(0, -1,0), (0, 1,0), (-1, 0,0), (1, 0,0),(0,0,1),(0,0,-1)]
    connected26=[(1,1,1),(1,1,0),(1,1,-1),
                                (1, 0,1),(1,0,0),(1,0,-1),
                                (1,-1,1),(1,-1,0),(1,-1,-1),

                                (0,1,1),(0,1,0),(0,1,-1),
                                (0, 0,1),(0,0,-1),
                                (0,-1,1),(0,-1,0),(0,-1,-1),

                                (-1,1,1),(-1,1,0),(-1,1,-1),
                                (-1, 0,1),(-1,0,0),(-1,0,-1),
                                (-1,-1,1),(-1,-1,0),(-1,-1,-1)]
    labels=np.zeros_like(image)
    current_label=1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                if image[i,j,k]==1 and labels[i,j,k]==0:
                    label = current_label
                    current_label+=1
                    labels[i,j,k]=label
                    stack=[(i,j,k)]
                    while len(stack)>0:
                        x,y,z=stack.pop()
                        for dx,dy,dz in connected26:
                            x2, y2,z2 = x + dx, y + dy,z+dz
                            if x2 >= 0 and x2 < image.shape[0] and y2 >= 0 and y2 < image.shape[1] and z2 >= 0 and z2 < image.shape[2] and image[x2, y2,z2] == 1 and labels[x2, y2,z2] == 0:
                                labels[x2, y2,z2] = label
                                stack.append((x2, y2,z2))
    return labels

def shortest_distance3D(labels,connect1,connect2):
    coords1 = np.transpose(np.where(labels == connect1))
    coords2 = np.transpose(np.where(labels == connect2))
    min_dist = float('inf')
    closest_coord1 = None
    closest_coord2 = None
    for coord1 in coords1:
        dists = np.linalg.norm(coords2 - coord1, axis=1)
        min_dist_for_coord1 = np.min(dists)
        if min_dist_for_coord1 < min_dist:
            min_dist = min_dist_for_coord1
            closest_coord1 = coord1
            closest_coord2 = coords2[np.argmin(dists)]
    for coord2 in coords2:
        dists = np.linalg.norm(coords1 - coord2, axis=1)
        min_dist_for_coord2 = np.min(dists)
        if min_dist_for_coord2 < min_dist:
            min_dist = min_dist_for_coord2
            closest_coord1 = coords1[np.argmin(dists)]
            closest_coord2 = coord2

    return min_dist, closest_coord1, closest_coord2
def connect_near3D(image, start, end, r1, r2):
    import math
    z1, y1, x1 = start
    z2, y2, x2 = end
    r = int(math.ceil((r1 + r2) / 2))
    if r == 0:
        r = 1
    x, y, z = x1, y1, z1
    xs = 1 if x1 < x2 else -1
    ys = 1 if y1 < y2 else -1
    zs = 1 if z1 < z2 else -1
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dz = abs(z1 - z2)
    path = []
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x != x2:
            x += xs
            if p1 >= 0:
                y += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz
            path.append([z, y, x])
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y != y2:
            y += ys
            if p1 >= 0:
                x += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            path.append([z, y, x])
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z != z2:
            z += zs
            if p1 >= 0:
                y += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            path.append([z, y, x])
    #print(path)
    for i in range(len(path)):
        for z in range(path[i][0]-r,path[i][0]+r+1):
            for y in range(path[i][1]-r,path[i][1]+r+1):
                for x in range(path[i][2]-r,path[i][2]+r+1):
                    image[z,y,x]=1
    return image
    
import SimpleITK as sitk
import math
def get_radius(img,x):
    for i in range(1,10):
        square=img[x[0]-i:x[0]+i+1,x[1]-i:x[1]+i+1,x[2]-i:x[2]+i]
        if np.count_nonzero(square)<(((2*i+1)**3)/2):
            break
    return i-1
def get_weight(image):
    weight=np.zeros((image.max(),image.max(),7))
    for i in range(image.max()):
        for j in range(i+1,image.max()):
            min_score,closest_coord1,closest_coord2=shortest_distance3D(image,i+1,j+1)
            weight[i,j,:]=[min_score,closest_coord1[0],closest_coord1[1],closest_coord1[2],closest_coord2[0],closest_coord2[1],closest_coord2[2]]
    return weight
def find_min(matrix):
    matrix=matrix[:,:,0]
    min_val =float('inf') 
    for i in range(0,len(matrix)):
        for j in range(i+1, len(matrix[0])):
            if matrix[i][j] < min_val:
                min_val = matrix[i][j]
                x,y=i,j
    return x,y
def shortest_distance3D(labels,directions,connect1,connect2):
    coords1 = np.transpose(np.where(labels == connect1))
    coords2 = np.transpose(np.where(labels == connect2))
    min_score = float('inf')
    closest_coord1 = None
    closest_coord2 = None
    for coord1 in coords1:
        for coord2 in coords2:
            dist = np.linalg.norm(coord1 - coord2)
            min_angle=float('inf')
            for i in range(1,int(directions[coord1[0],coord1[1],coord1[2],0])+1):
                for j in range(1,int(directions[coord2[0],coord2[1],coord2[2],0])+1):
                    #print(directions[coord1[0],coord1[1],coord1[2],:])
                    a=np.array(directions[coord1[0],coord1[1],coord1[2],(i-1)*3+1:(i-1)*3+4])
                    b=np.array(directions[coord2[0],coord2[1],coord2[2],(j-1)*3+1:(j-1)*3+4])
                    c=np.array([coord1[0],coord1[1],coord1[2]])-np.array([coord2[0],coord2[1],coord2[2]])
                    angle1=np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
                    angle2=np.dot(a, c)/(np.linalg.norm(a)*np.linalg.norm(c))
                    angle3=np.dot(b, c)/(np.linalg.norm(b)*np.linalg.norm(c))
                    angle=np.sqrt(1-angle1**2)+np.sqrt(1-angle2**2)+np.sqrt(1-angle3**2)
                    #print(a,b,c,angle1,angle2,angle3,angle)
                    if angle<min_angle:
                        min_angle=angle
            score = dist*(min_angle+1)
            if score < min_score:
                min_score = score
                closest_coord1 = coord1
                closest_coord2 = coord2
    return min_score,closest_coord1,closest_coord2
def get_direction3D(image):
    directions=np.zeros((image.shape[0],image.shape[1],image.shape[2],3*26+1))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                if image[i,j,k]!=0:
                    nearpoint=get_nearpoint3D(image,i,j,k)
                    if nearpoint.shape[0]!=0:
                        directions[i,j,k,0]=nearpoint.shape[0]
                        m=1
                        #print(nearpoint.shape[0])
                        for n in range(nearpoint.shape[0]):
                            directions[i,j,k,m]=nearpoint[n,0]-i
                            directions[i,j,k,m+1]=nearpoint[n,1]-j
                            directions[i,j,k,m+2]=nearpoint[n,2]-k
                            m+=3
                    #print(directions[i,j,k,:])
    return directions
def get_weight(image,directions):
    weight=np.zeros((image.max(),image.max(),7))
    for i in range(image.max()):
        for j in range(i+1,image.max()):
            #print(i,j)
            min_score,closest_coord1,closest_coord2=shortest_distance3D(image,directions,i+1,j+1)
            weight[i,j,:]=[min_score,closest_coord1[0],closest_coord1[1],closest_coord1[2],closest_coord2[0],closest_coord2[1],closest_coord2[2]]
    return weight
def get_nearpoint3D(image,x,y,z):
    neighbors=np.empty((0, 3))
    for dx in range(-1,2):
        for dy in range(-1,2):
            for dz in range(-1,2):
                if dx==0 and dy==0 and dz==0:
                        continue
                nx, ny ,nz= int(x + dx), int(y + dy),int (z+dz)
                if nx >= 0 and nx < image.shape[0] and ny >= 0 and ny < image.shape[1] and nz >= 0 and nz < image.shape[2]:
                    if(image[nx,ny,nz]!=0):
                        neighbors=np.append(neighbors,np.array([nx,ny,nz]).reshape(1, 3),axis=0)
    return neighbors
img=sitk.ReadImage('network_prediction.seg.nrrd')
image=sitk.GetArrayFromImage(img)
image=np.where(image!=0,1,0)
outputimg=image
while(1):
    sk=morphology.skeletonize(outputimg)
    sk=np.where(sk!=0,1,0)
    sk_count=connected_components3D(sk)
    if(sk_count.max()==1):
        break
    n_sk_count=np.zeros(sk_count.max())
    print(sk_count.max())
    weights=np.zeros((sk_count.max(),sk_count.max(),7))

    directions=get_direction3D(sk)
    
    weights=get_weight(sk_count,directions)
    # if np.all(weights[weights != 0] > 200):
    #     break
    print(weights[:,:,0])
    x,y=find_min(weights)
    start=[int(weights[x,y,1]),int(weights[x,y,2]),int(weights[x,y,3])]
    end=[int(weights[x,y,4]),int(weights[x,y,5]),int(weights[x,y,6])]
    r1,r2=get_radius(img,start),get_radius(img,end)
    #print(x+1,y+1,start,end)
    sk=connect_near3D(sk,start,end,r1,r2)
    outputimg+=sk
    outputimg=np.where(outputimg!=0,1,0)
outputSeg = sitk.GetImageFromArray(outputimg.astype(np.uint8), isVector=False)
outputSeg.SetSpacing(img.GetSpacing())
outputSeg.SetOrigin(img.GetOrigin())
outputSeg.SetDirection(img.GetDirection())
sitk.WriteImage(outputSeg, 'result.seg.nrrd', True)

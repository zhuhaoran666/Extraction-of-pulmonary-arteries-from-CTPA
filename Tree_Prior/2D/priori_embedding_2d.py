import cv2
import numpy as np
from matplotlib.pyplot import *
from skimage import morphology
from scipy.ndimage import distance_transform_edt
import math
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
def connected_components2D(image):
    labels = np.zeros_like(image, dtype=np.int32)
    current_label = 1
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 1 and labels[i, j] == 0:
                label = current_label
                current_label += 1
                labels[i, j] = label
                stack = [(i, j)] 
                while len(stack) > 0:
                    x, y = stack.pop()
                    connected4=[(0, -1), (0, 1), (-1, 0), (1, 0)]
                    connected8=[(0, -1), (0, 1), (-1, 0), (1, 0),(-1,-1),(1,1),(1,-1),(-1,1)]
                    for dx, dy in connected8:
                        x2, y2 = x + dx, y + dy
                        if x2 >= 0 and x2 < image.shape[0] and y2 >= 0 and y2 < image.shape[1] and image[x2, y2] == 1 and labels[x2, y2] == 0:
                            labels[x2, y2] = label
                            stack.append((x2, y2))
    return labels
def shortest_distance(labels,directions,connect1,connect2):
    coords1 = np.transpose(np.where(labels == connect1))
    coords2 = np.transpose(np.where(labels == connect2))
    min_score = float('inf')
    closest_coord1 = None
    closest_coord2 = None
    for coord1 in coords1:
        for coord2 in coords2:
            dist = np.linalg.norm(coord1 - coord2)
            min_angle=float('inf')
            for i in range(1,int(directions[coord1[0],coord1[1],0])+1):
                for j in range(1,int(directions[coord2[0],coord2[1],0])+1):
                    angle1=np.sin(abs(directions[coord1[0],coord1[1],i]-directions[coord2[0],coord2[1],j]))
                    angle2=np.sin(abs(np.arctan2(coord1[0]-coord2[0],coord1[1]-coord2[1])%np.pi-directions[coord1[0],coord1[1],i]))
                    angle3=np.sin(abs(np.arctan2(coord1[0]-coord2[0],coord1[1]-coord2[1])%np.pi-directions[coord2[0],coord2[1],j]))
                    angle=angle1+angle2+angle3
                    if angle<min_angle:
                        min_angle=angle
            score = dist*(min_angle+1)
            if score < min_score:
                min_score = score
                closest_coord1 = coord1
                closest_coord2 = coord2
    return min_score,closest_coord1,closest_coord2
def get_weight(image,directions):
    weight=np.zeros((image.max(),image.max(),5))
    for i in range(image.max()):
        for j in range(i+1,image.max()):
            #print(i,j)
            min_score,closest_coord1,closest_coord2=shortest_distance(image,directions,i+1,j+1)
            weight[i,j,:]=[min_score,closest_coord1[0],closest_coord1[1],closest_coord2[0],closest_coord2[1]]
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
def get_direction(image):
    directions=np.zeros((image.shape[0],image.shape[1],9))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j]!=0:
                nearpoint=get_nearpoint(image,i,j)
                if nearpoint.shape[0]!=0:
                    if nearpoint.shape[0]==1:
                        directions[i,j,0]=1
                        directions[i,j,1]=np.arctan2(nearpoint[0,0]-i,nearpoint[0,1]-j)%np.pi
                    if nearpoint.shape[0]==2:
                        directions[i,j,0]=1
                        directions[i,j,1]=np.arctan2(nearpoint[0,0]-nearpoint[1,0],nearpoint[0,1]-nearpoint[1,1])%np.pi
                    if nearpoint.shape[0]>2:
                        directions[i,j,0]=nearpoint.shape[0]
                        k=1
                        for n in range(nearpoint.shape[0]):
                            directions[i,j,k]=np.arctan2(nearpoint[n,0]-i,nearpoint[n,1]-j)%np.pi
                            k+=1
    return directions
def get_nearpoint(image,x,y):
    neighbors=np.empty((0, 2))
    for dx in range(-1,2):
        for dy in range(-1,2):
            if dx==0 and dy==0:
                    continue
            nx, ny = int(x + dx), int(y + dy)
            if nx >= 0 and nx < image.shape[0] and ny >= 0 and ny < image.shape[1]:
                if(image[nx,ny]!=0):
                    neighbors=np.append(neighbors,np.array([nx,ny]).reshape(1, 2),axis=0)
    return neighbors
def get_radius(img,x):
    for i in range(1,10):
        square=img[x[0]-i:x[0]+i+1,x[1]-i:x[1]+i+1]
        if np.count_nonzero(square)<((2*i+1)**2):
            break
    return i-1
def connect_near(image,start,end,r1,r2):
    y1, x1 = start
    y2, x2 = end
    r=int(math.ceil((r1+r2)/2))
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = x1, y1
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while x != x2 or y != y2:
        for i in range(y-r,y+r+1):
            for j in range(x-r,x+r+1):
                image[i, j] = image[y1,x1]
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        # if x<0 or x>=image.shape[1] or y<0 or y>=image.shape[0]:
        #     break
    return image

img=cv2.imread('network_prediction.png')[:,:,0]
img=np.where(img!=0,1,0)
sk=morphology.skeletonize(img,method='lee').astype(np.uint16)
while(1):
    sk_count=connected_components2D(sk)
    if(sk_count.max()==1):
        break
    n_sk_count=np.zeros(sk_count.max())
    weights=np.zeros((sk_count.max(),sk_count.max(),7))
    directions=get_direction(sk)
    weights=get_weight(sk_count,directions)
    x,y=find_min(weights)
    print(weights[:,:,0])
    start=[int(weights[x,y,1]),int(weights[x,y,2])]
    end=[int(weights[x,y,3]),int(weights[x,y,4])]
    r1=get_radius(img,start)
    r2=get_radius(img,end)
    #print(r1,r2)
    sk=connect_near(sk,start,end,r1,r2)
kernel = np.ones((3, 3), dtype=np.uint8)
dilated_img = binary_dilation(sk, kernel)
dilated_img =np.logical_or(sk, img).astype(int)
cv2.imwrite('result.png',dilated_img*255)
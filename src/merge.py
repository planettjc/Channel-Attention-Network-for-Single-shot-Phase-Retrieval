import tifffile
from PIL import Image
import numpy as np
import cv2
from imageio import imsave
import os
patchsize = 256
effectivelen = 180 #effective coordinate: range(n*effectivelen+(patchsize-effectivelen)/2,(n+1)*effectivelen+(patchsize-effectivelen)/2)
padding = int((patchsize-effectivelen)/2)

sp = [2044,2048]
xlist = list(range(0,sp[0]-2*patchsize,effectivelen))+[sp[0]-2*patchsize,sp[0]-2*patchsize-effectivelen]
ylist = list(range(0,sp[1]-2*patchsize,effectivelen))+[sp[1]-2*patchsize,sp[1]-2*patchsize-effectivelen]
i = 10
arr = np.zeros((sp[0],sp[1]),dtype=np.dtype('int32')).astype('uint16')
for k,kk in enumerate(xlist):
    for l,ll in enumerate(ylist):
        tmp = np.array(Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../experiment','phase_test_result','results-Test','%d'%i+'%02d'%k+'%02d'%l+'_PH_Recon.png')))
        if k == 0:
            up = 0
        else:
            up = padding
        if l == 0:
            lt = 0
        else:
            lt = padding
        if k < len(xlist) - 1 and l < len(ylist) - 1:
            arr[kk+up:kk+padding+effectivelen,ll+lt:ll+padding+effectivelen] = tmp[up:patchsize-padding,lt:patchsize-padding]
        if k == len(xlist) - 1:
            dw = padding
        else:
            dw = 0
        if l == len(ylist) - 1:
            rt = padding
        else:
            rt = 0
        if k > len(xlist) - 3 or l > len(ylist) - 3:
            arr[kk+patchsize+padding:kk+2*patchsize-dw,ll+patchsize+padding:ll+2*patchsize-rt] = tmp[patchsize+padding:2*patchsize-dw,patchsize+padding:2*patchsize-rt]
            if k <= 1:
                arr[kk+up:kk+padding+effectivelen,ll+patchsize+padding:ll+2*patchsize-rt] = tmp[up:patchsize-padding,patchsize+padding:2*patchsize-rt]
            if l <= 1:
                arr[kk+patchsize+padding:kk+2*patchsize-dw,ll+lt:ll+padding+effectivelen] = tmp[patchsize+padding:2*patchsize-dw,lt:patchsize-padding]
imsave(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../experiment','phase_test_result','Merged_'+'%d'%i+'.tif'),arr.astype('uint16'))
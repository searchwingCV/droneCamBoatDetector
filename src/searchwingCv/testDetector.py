#!/usr/bin/env python

import sys
import cv2
import time
cv2.useOptimized()

import roiDetector

if len(sys.argv)>1:
    path = sys.argv[1]
else:
    print("Set imagename as argument! ... abort.")
    exit(0)

#load
pic=roiDetector.load_pic(path)

#process
print("Run ROI detector one time for warmup...")
doGaussBlur=True
ROIs, contours, mask_img = roiDetector.detectROIs(pic, doGaussBlur=doGaussBlur, gaussBlurKernelSize=5, gradSize=3, gradThreshold=0.9994, openSize=2)

print("Run ROI detector 5 times...")
for i in range(5):
    start = time.time()
    ROIs, contours, mask_img = roiDetector.detectROIs(pic, doGaussBlur=doGaussBlur, gaussBlurKernelSize=5, gradSize=3, gradThreshold=0.9994, openSize=2)
    end = time.time()
    print("roiDetector duration [sek]:", end - start)

#visualize
imgBoundingBoxes=roiDetector.drawBoundingBoxesToImg(pic,ROIs)
cv2.imwrite("testOut.jpg", imgBoundingBoxes)

#plot
#import matplotlib.pyplot as plt
#out = roiDetector.drawBoundingBoxesToImg(pic, ROIs)
#plt.imshow(out,cmap="gray")
#plt.show()

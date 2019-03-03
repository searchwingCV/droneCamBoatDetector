#!/usr/bin/env python

import sys
import cv2
import time
cv2.useOptimized()
import matplotlib.pyplot as plt

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

ROIs, contours, mask_img = roiDetector.detectROIs(pic, gradSize=1, gaussBlurKernelSize=15, gradThreshold=99.4,
                                                  openSize=3)

print("Run ROI detector 5 times...")
for i in range(10):
    start = time.time()
    ROIs, contours, mask_img = roiDetector.detectROIs(pic, gradSize=1, gaussBlurKernelSize=15, gradThreshold=99.4,
                                                      openSize=3)
    end = time.time()
    print("roiDetector duration [sek]:", end - start)
#visualize
out = roiDetector.drawBoundingBoxesToImg(pic, ROIs)
plt.imshow(out,cmap="gray")
plt.show()
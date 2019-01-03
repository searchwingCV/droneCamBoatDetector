####################################################################################
#  Detect ROIs for all images in a given path and save the ROIs as a XML-File in the PASCAL Voc Format
#####################################################################################

import cv2
import numpy as np;
import matplotlib.pyplot as plt
cv2.useOptimized()
import time
import os
from lxml import etree as ET
import os.path

from searchwing import roiDetector
#Change this path to the desired folder to export labels in pascal voc format
path=""

def export2PascalVocXml(boundingBoxes,filepath,inPic):
    imgPathFull = filepath 
    imgFolderPath = os.path.dirname(imgPathFull)
    imgFolderName = os.path.basename(os.path.dirname(onePicPath))
    imgFileName = os.path.basename(imgPathFull)

    root = ET.Element('annotation')

    folder = ET.SubElement(root, 'folder')
    folder.text = imgFolderName
    filename = ET.SubElement(root, 'filename')
    filename.text = imgFileName
    path = ET.SubElement(root, 'path')
    path.text = imgPathFull

    source = ET.SubElement(root, 'source')
    database = ET.SubElement(source, 'Type')
    database.text = 'Unknown'

    size = ET.SubElement(root, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(inPic.shape[0])
    height = ET.SubElement(size, 'height')
    height.text = str(inPic.shape[1])
    depth = ET.SubElement(size, 'depth')
    depth.text = str(inPic.shape[2])

    segmented = ET.SubElement(root, 'segmented')
    segmented.text = str(0)

    for bb in boundingBoxes:
        newObj = ET.SubElement(root, 'object')
        newObj_name = ET.SubElement(newObj, 'name')
        newObj_name.text = "nature"
        newObj_pose = ET.SubElement(newObj, 'pose')
        newObj_pose.text = "Unspecified"
        newObj_truncated = ET.SubElement(newObj, 'truncated')
        newObj_truncated.text = "0"
        newObj_difficult = ET.SubElement(newObj, 'difficult')
        newObj_difficult.text = "0"

        newObj_bndbox = ET.SubElement(newObj, 'bndbox')
        newObj_bndbox_xmin = ET.SubElement(newObj_bndbox, 'xmin')
        newObj_bndbox_xmin.text = str(bb[0])
        newObj_bndbox_ymin = ET.SubElement(newObj_bndbox, 'ymin')
        newObj_bndbox_ymin.text = str(bb[1])
        newObj_bndbox_xmax = ET.SubElement(newObj_bndbox, 'xmax')
        newObj_bndbox_xmax.text = str(bb[2])
        newObj_bndbox_ymax = ET.SubElement(newObj_bndbox, 'ymax')
        newObj_bndbox_ymax.text = str(bb[3])

    tree = ET.ElementTree(root)
    print(imgPathFull)
    savePath = os.path.splitext(imgPathFull)[0] + '.xml'
    print(savePath)
    if os.path.isfile(savePath) == False:
        tree.write(savePath, pretty_print=True)
    else:
        print("File already exists... therefor i am not going to overwrite it. (to help dumb users, for example MYSELF)")
    

# Get files
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) and "xml" not in f]
filesWithPath = []
for f in onlyfiles:
    filesWithPath.append(path+f)

# Detect ROIs and save those alongside them as a XML-File in the PASCAL Voc Format
for onePicPath in filesWithPath:
    print(onePicPath)
    onePic=roiDetector.load_pic(onePicPath)
    boundingBoxes,_,_=roiDetector.detectROIs(onePic)
    export2PascalVocXml(boundingBoxes,onePicPath,onePic)


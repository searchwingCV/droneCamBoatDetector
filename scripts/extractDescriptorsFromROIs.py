####################################################################################
# Extract descriptors to train classifier 
# You need to provide a folder with images and corresponding labels in the pascal voc format
#####################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt
cv2.useOptimized()
from lxml import etree as ET
import time
import pandas as pd

from searchwing import roiDetector
from searchwing import roiDescriptor

#Folder with the images and labels (from the downloaded zip file)
pathIn = ""
#Filepath to output the descriptors as hdf5 file
filePathOut = "/extracedDescriptors.h5"

def getDescriptorsFromFile(path):
    descriptors=[]
    #load xml
    tree = ET.parse(path)
    #get img path
    imgFilePath=tree.find("path").text

    #get ROIs
    ROIs=[]
    ROIs=getAnnotationsFromXml(tree)

    img_bgr = cv2.imread(imgFilePath, cv2.IMREAD_COLOR) 
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(imgFilePath)
    _,_,img_mask=roiDetector.detectROIs(img_rgb)
    #plt.imshow(img_mask)
    #plt.show()
    
    print(len(ROIs))
    start=time.time()
    for ROI in ROIs:
            
        width=ROI[3]-ROI[1]
        height=ROI[4]-ROI[2]
        pixcount=width*height
        
        img_bgr_roi = img_bgr[ROI[2]:ROI[4], ROI[1]:ROI[3]]
        img_mask_roi = img_mask[ROI[2]:ROI[4], ROI[1]:ROI[3]]
        
        filterTmp=cv2.findNonZero(img_mask_roi)
        if filterTmp is None:
            continue
            
        oneDescr=descriptor.calcDescrROI(img_bgr_roi,img_mask_roi)
        metaInfo = np.array((imgFilePath,width,height,pixcount,ROI[1],ROI[2],ROI[3],ROI[4],ROI[0]))
        descrMeta = np.hstack((metaInfo,oneDescr))
        descriptors.append(descrMeta)
        
    end = time.time()
    print("Processingduration [sek]:",end - start)
    return descriptors
    

# Get xml files
import os
def getAnnotationFilePaths(path):
    onlyfiles = [f for f in os.listdir(path) if (os.path.splitext(f)[1] == ".xml") ]
    filePaths = [path + f for f in onlyfiles]
    return filePaths

def getAnnotationsFromXml(xmlData):
    objects=[]
    for Obj in xmlData.iterfind("object"):
        Obj_name = Obj.find("name").text
        Obj_bndbox=Obj.find("bndbox")
        Obj_bndbox_xmin=Obj_bndbox.find("xmin").text
        Obj_bndbox_ymin=Obj_bndbox.find("ymin").text
        Obj_bndbox_xmax=Obj_bndbox.find("xmax").text
        Obj_bndbox_ymax=Obj_bndbox.find("ymax").text
        oneObj = [Obj_name,int(Obj_bndbox_xmin),int(Obj_bndbox_ymin),int(Obj_bndbox_xmax),int(Obj_bndbox_ymax)]
        objects.append(oneObj)
    return objects


print("I am going to read all xml files from " +pathIn + " and calculate the descriptor for it")
print("The output is going to be saved as hdf5 in "+filePathOut)

descriptor = roiDescriptor.Descriptors()
descriptorNames=descriptor.getDescrNames()

MetaDataNames=np.array(("imgFilePath","width","height","pixcount","xmin","ymin","xmax","ymax","class"))
colnames=np.hstack((MetaDataNames,descriptorNames))

filePaths=getAnnotationFilePaths(pathIn)
print("Start extraction of descriptors ...")

df = pd.DataFrame(columns=colnames)
for onePath in filePaths[:]:
    print(onePath)
 
    desc=getDescriptorsFromFile(onePath)
    
    #save descriptor to hdf5
    if(len(desc)>0):
        descVStack=np.vstack(desc)
        df=pd.DataFrame(descVStack,columns=colnames)
        df=df.convert_objects(convert_numeric=True)
        df.to_hdf(filePathOut,
                  key="table",
                  mode="a",
                  append=True,
                  dropna=False,
                  format="table") #use append to avoid huge memoryconsumption


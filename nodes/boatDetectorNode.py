#!/usr/bin/env python
####################################################################################
# The central ROS node to receive all datastreams
# It receives images from the camera
# It outputs 3D-position estimates of the detected boats
####################################################################################
import cv2
cv2.useOptimized()
import numpy as np
import time
import sys,os

#ros stuff
import rospy
import tf
from sensor_msgs.msg import Image
from image_geometry import cameramodels
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

#searchwing stuff
from searchwingCv import roiDetector
from searchwingCv import roiTracker

rospy.init_node('roiDetector', anonymous=True)
from searchwingCvRos import debugVis
from searchwingCvRos import imagePointTransformations
from searchwingCvRos import camCalibParser

#####################
# IO of the ROS-Node
#####################
#Entry point of the programm to receive data through subscribers
#Program is eventtriggered, thus it calls the callback when a new Image is received
def listener():
    imgTopicName = 'camera/image_raw' # subscribe the camera data stream
    rospy.Subscriber(imgTopicName,Image,callbackImg)

    ##Not used currently as we load the calib from file directly
    #rospy.Subscriber(calibTopicName,CameraInfo,callbackCamCalib)
    #calibTopicName = "camera/camera_info" # subscribe to the camera calib data stream

    print("node started: Loop until new data arrives")
    #Loop until new data arrives
    rospy.spin()

#####################
# Init
#####################
dbgVis = debugVis.dbgVis()
cvbridge = CvBridge()
dirname, filename = os.path.split(sys.argv[0])

#Get cam intrinsic calibration
camModel = cameramodels.PinholeCameraModel()
intrinsicCalibPath = dirname + "/../config/bodenseeCamCalibResized.yaml"
camIntrinsicCamInfo = camCalibParser.getCameraIntrinsicCalib(intrinsicCalibPath)
camModel.fromCameraInfo(camIntrinsicCamInfo)
dbgVis.setCamIntrinsics(camModel)

#Transform functions
tf_listener = tf.TransformListener() # must be outside callback

tracker = roiTracker.roiTracker()

#####################
# Process datastreams
#####################

#The central callbackfunction which is called when a image provided through the camera/image_raw topic/datastream
def callbackImg(data):
    if camModel.K is None:  # Only continue if cam calibration is received
        return
    picStamp = data.header.stamp #important to get the timestamp of the recording of the image to get the 3d-position of the drone at the time

    startAll = time.time()

    #Get Picture
    start = time.time()
    cv_img = cvbridge.imgmsg_to_cv2(data, "bgr8")
    rgb=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
    end = time.time()
    print("convert pic [sek]:",end - start)

    ####Run ROI-Detection Pipeline
    start = time.time()
    ROIs,contours,mask_img=roiDetector.detectROIs(rgb, doGaussBlur=False, gaussBlurKernelSize=5, gradSize=3, gradThreshold=0.9994, openSize=2)
    end = time.time()
    imgDbgVis = roiDetector.drawBoundingBoxesToImg(rgb,ROIs)
    print("detectROIs [sek]:",end - start)

    ####Get 3D-Position of drone in the World
    start = time.time()
    drone3dPos = PointStamped()
    drone3dPos.point.x = 0
    drone3dPos.point.y = 0
    drone3dPos.point.z = 0
    drone3dPos.header.frame_id = "base_link"
    drone3dPos.header.stamp = picStamp
    tf_listener.waitForTransform("base_link","map",picStamp,rospy.Duration(1)) #wait until the needed transformation is ready, as the image can be provided faster than the telemetry
    drone3dPosInMap=tf_listener.transformPoint("map",drone3dPos)

    ####Filtering of ROIs
    min3DLen = 2  # [m]
    max3DLen = 35  # [m]
    min2DSize = 13  # [pix]
    max2DSize = 150  # [pix]
    min2dArea = 13 * 13

    contoursSizeFiltered=[]
    ROIsSizeFiltered=[]
    ROIs3dCenters=[]
    # Filter by 2D-Size of ROIs
    for idx, oneContour in enumerate(ROIs, 0):
        ROIwidth = ROIs[idx][2]-ROIs[idx][0]
        ROIheight = ROIs[idx][3]-ROIs[idx][1]
        ROIarea= ROIwidth*ROIheight
        if ROIwidth < min2DSize or ROIwidth > max2DSize or ROIheight < min2DSize or ROIheight > max2DSize or ROIarea < min2dArea:
            continue
        pt2dROICenter = np.array([ROIs[idx][0] + (ROIwidth / 2) , ROIs[idx][1] + (ROIheight / 2)], np.int)
        pt3dROICenter = imagePointTransformations.getPixel3DPosOnWater(pt2dROICenter, drone3dPosInMap, camModel, picStamp)  # Get 3D Position of
        contoursSizeFiltered.append(oneContour)
        ROIsSizeFiltered.append(ROIs[idx])
        ROIs3dCenters.append(pt3dROICenter)

    """
    #Filter by estimated 3D-Size of Objects in ROIs
    for idx, oneContour in enumerate(contours, 0):
        pt2D1,pt2D2=roiDetectorCV.getExtreme2dPoints(oneContour,debug=False)
        pt3D1 = getPixel3DPosOnWater(pt2D1,drone3dPosInMap,picStamp)
        pt3D2 = getPixel3DPosOnWater(pt2D2,drone3dPosInMap,picStamp)
        pt3DDiff = pt3D1-pt3D2
        objLen3D = np.linalg.norm(pt3DDiff)
        pt3dROICenter = pt3D2 + pt3DDiff*0.5
        if objLen3D > min3DLen and objLen3D < max3DLen:
            contoursSizeFiltered.append(oneContour)
            ROIsSizeFiltered.append(ROIs[idx])
            ROIs3dCenters.append(pt3dROICenter)
            #print(oneContour[0],dist3D)
            cv2.line(imgBB, (pt2D1[0],pt2D1[1]),(pt2D2[0],pt2D2[1]),(0, 255, 0), 3)
    """
    end = time.time()
    print("3d pos estimate [sek]:", end - start)

    valid3DDetections=[]
    for oneROI3DPt in ROIs3dCenters:
        valid3DDetections.append(oneROI3DPt)

    #Track roiDetections over time to validate them
    start = time.time()
    associationTresh = 40#[m]
    tracker.addDetections(valid3DDetections,associationTresh)
    trackedBoats=[]
    trackedBoats=tracker.getTrackings(minTrackingCount=3) # all tracked objects which got tracked >= minTrackingCount-times are considered as boats
    for oneBoat in trackedBoats:
        print("Tracker found boat: id ",oneBoat.id," trackCount ",oneBoat.trackingCount," lifetime ",oneBoat.lifetime);
    end = time.time()
    print("Tracker[sek]:",end - start)

    #DbgVis
    start = time.time()
    dbgVis.createDbgVisImage(imgDbgVis,ROIsSizeFiltered,tracker,picStamp)
    dbgVis.createImageEdges(imgDbgVis,drone3dPosInMap,picStamp)
    dbgVis.createPlanePath(imgDbgVis,drone3dPosInMap,picStamp)
    end = time.time()
    print("DbgVis[sek]:",end - start)

    endAll = time.time()
    print("================================== sum of all [sek]:",endAll - startAll)




if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

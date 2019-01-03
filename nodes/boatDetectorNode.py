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
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from image_geometry import cameramodels
import tf
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
import sys,os

from searchwing import roiDetector
from searchwing import roiDescriptor
from searchwing import roiTracker
from searchwing import camCalib

cvbridge = CvBridge()
dirname, filename = os.path.split(sys.argv[0])
#####################
# IO of the ROS-Node
#####################
rospy.init_node('roiDetector', anonymous=True)
#Entry point of the programm to receive data through subscribers
#Program is eventtriggered, thus it calls the callback when a new Image is received
def listener():
    imgTopicName = 'camera/image_raw' # subscribe the camera data stream
    calibTopicName = "camera/camera_info" # subscribe to the camera calib data stream
    rospy.Subscriber(imgTopicName,Image,callbackImg)
    #rospy.Subscriber(calibTopicName,CameraInfo,callbackCamCalib)
    print("node started: Loop until new data arrives")
    #Loop until new data arrives
    rospy.spin()

#Callback for the camera calib which is provided from external ros node
camModel = cameramodels.PinholeCameraModel()
intrinsicCalibPath = dirname + "/../config/bodenseeCamCalibResized.yaml"
camIntrinsicCamInfo = camCalib.getCameraIntrinsicCalib(intrinsicCalibPath)
camModel.fromCameraInfo(camIntrinsicCamInfo)

#Debugvisualization Publishers
import sensor_msgs.point_cloud2 as pc2
debugPicPub = rospy.Publisher('detectionsPic', Image, queue_size=1)
picProjPub = rospy.Publisher('picProjection',PointCloud2, queue_size=1)
planePathPub = rospy.Publisher('planePath',Path, queue_size=1)
planePath = Path()
planePath.header.frame_id = "map"

#####################
# Process datastreams
#####################

#Transform functions
tf_listener = tf.TransformListener() # must be outside callback
tf_transformer = tf.Transformer()
tf_broadcaster = tf.TransformBroadcaster()

#Stuff to load the ROI descriptor
descriptor = roiDescriptor.Descriptors()
descriptorLen = len(descriptor.getDescrNames())
#Stuff to load the classifier
from sklearn.externals import joblib
import rospkg
rospack = rospkg.RosPack()
classifierpath = dirname + "/../config/classifier.pkl"
clf=joblib.load(classifierpath)
clf.n_jobs=1

tracker = roiTracker.roiTracker()

#######Helper Functions
# A generic function to compute the intersection of a 3D Line with a 3D Plane
def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
	    raise RuntimeError("no intersection or line is within plane")
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi

# Function to get the 3D Position of a Pixel on the Water in a Picture by transforming the 2D Pos of the Pixel to a 3D Pos in the World
def getPixel3DPosOnWater(Point2D,Drone3DPosInMap,stamp):
    #get Vector from origin to 3d pos of pixel in the camcoordinatesystem
    #see https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html for visualization
    Point3D = camModel.projectPixelTo3dRay(Point2D)

    #generate point in cam coordinate system in ros
    pt = PointStamped()
    pt_transformed = PointStamped()
    pt.point.x = Point3D[0]
    pt.point.y = Point3D[1]
    pt.point.z = Point3D[2]
    pt.header.frame_id = "cam"
    pt.header.stamp = stamp

    #transform point to drone coordinate system
    pt_transformed = tf_listener.transformPoint("map", pt)

    # Define plane on ground = sea
    planeNormal = np.array([0, 0, 1])
    planePoint = np.array([0, 0, 0])

    # Define ray through pixel in drone coordinate system
    rayPoint = np.array([Drone3DPosInMap.point.x,
                         Drone3DPosInMap.point.y,
                         Drone3DPosInMap.point.z])  # Any point along the ray
    rayDirection=np.array([pt_transformed.point.x-Drone3DPosInMap.point.x,
                           pt_transformed.point.y-Drone3DPosInMap.point.y,
                           pt_transformed.point.z-Drone3DPosInMap.point.z])
    rayPointOnPlane = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
    return rayPointOnPlane

# Function to get the pixelpos of a 3D Pos in the World
def project3DPosToPic(Point3D,camModel,stamp):
    pt = PointStamped()
    pt.point.x = Point3D[0]
    pt.point.y = Point3D[1]
    pt.point.z = Point3D[2]
    pt.header.frame_id = "map"
    pt.header.stamp = stamp
    #transform point to cam coordinate system
    Point3DinCamCoord = tf_listener.transformPoint("cam", pt)
    Point2DinPicCoord = camModel.project3dToPixel([Point3DinCamCoord.point.x,Point3DinCamCoord.point.y,Point3DinCamCoord.point.z])
    return Point2DinPicCoord

#The central callbackfunction
def callbackImg(data):
    if camModel.K is None:  # Only continue if cam calibration is received
        return
    picStamp = data.header.stamp #important to get the timestamp of the recording of the image to get the 3d-position of the drone at the time

    startAll = time.time()

    #Get Picture
    cv_img = cvbridge.imgmsg_to_cv2(data, "bgr8")
    rgb=cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)

    ####Run ROI-Detection Pipeline
    start = time.time()
    ROIs,contours,mask_img=roiDetector.detectROIs(rgb, gradSize=1, gaussBlurKernelSize=15, openSize=3)
    imgDbgVis = roiDetector.drawBoundingBoxesToImg(rgb,ROIs)
    end = time.time()
    print("detectROIs [sek]:",end - start)

    ####Get 3D-Position of drone in the World
    start = time.time()
    drone3dPos = PointStamped()
    drone3dPos.point.x = 0
    drone3dPos.point.y = 0
    drone3dPos.point.z = 0
    drone3dPos.header.frame_id = "base_link"
    drone3dPos.header.stamp = picStamp
    tf_listener.waitForTransform("base_link","map",picStamp,rospy.Duration(1)) #wait until the needed transformation is ready, as the image can be provided faster the the telemetry
    drone3dPosInMap=tf_listener.transformPoint("map",drone3dPos)

    ####Filtering of ROIs
    min3DLen = 2  # [m]
    max3DLen = 35  # [m]
    min2DSize = 13  # [pix]
    max2DSize = 80  # [pix]
    min2dArea = 13 * 13

    contoursSizeFiltered=[]
    ROIsSizeFiltered=[]
    ROIs3dCenters=[]
    # Filter by 2D-Size of ROIs
    for idx, oneContour in enumerate(contours, 0):
        ROIwidth = ROIs[idx][2]-ROIs[idx][0]
        ROIheight = ROIs[idx][3]-ROIs[idx][1]
        ROIarea= ROIwidth*ROIheight
        if ROIwidth < min2DSize or ROIwidth > max2DSize or ROIheight < min2DSize or ROIheight > max2DSize or ROIarea < min2dArea:
            continue
        pt2dROICenter = np.array([ROIs[idx][0] + (ROIwidth / 2) , ROIs[idx][1] + (ROIheight / 2)], np.int)
        pt3dROICenter = getPixel3DPosOnWater(pt2dROICenter, drone3dPosInMap, picStamp)  # Get 3D Position of
        contoursSizeFiltered.append(oneContour)
        ROIsSizeFiltered.append(ROIs[idx])
        ROIs3dCenters.append(pt3dROICenter)
        # print(oneContour[0],dist3D)
        #cv2.circle(imgDbgVis, (pt2dROICenter[0], pt2dROICenter[1]), 10, (0, 255, 0), 1)
        cv2.rectangle(imgDbgVis, (ROIs[idx][0], ROIs[idx][1]),
                      (ROIs[idx][2], ROIs[idx][3]), (255, 255, 0), 3)

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
    """
    #####Roi Descriptor
    #Get cutout images of the ROIs
    start = time.time()
    roiBGRImages = roiDetector.extractImagesFromROIs(ROIsSizeFiltered, cv_img)
    roiMASKImages = roiDetector.extractImagesFromROIs(ROIsSizeFiltered, mask_img)
    end = time.time()
    print("extractImagesFromROIs [sek]:", end - start)

    #Calculate descriptor of each cutout image
    roiDescriptions = np.empty((len(ROIsSizeFiltered), descriptorLen,))
    roiDescriptions[:] = np.nan
    start = time.time()
    i = 0
    for (roiBGR, roiMASK) in zip(roiBGRImages, roiMASKImages):
        description = descriptor.calcDescrROI(roiBGR, roiMASK)
        roiDescriptions[i, :] = description
        i = i + 1
    end = time.time()
    #Filter results on validity
    nonNanIndizes=~np.isnan(roiDescriptions).any(axis=1)
    npROIs=np.asarray(ROIsSizeFiltered)
    roiDescriptions=roiDescriptions[nonNanIndizes]
    validROIs=npROIs[nonNanIndizes]
    print("calcDescrROI [sek]:", end - start)

    ####Classify the descriptors of the ROIs
    start = time.time()
    roiClassifications=clf.predict(roiDescriptions)
    end = time.time()
    print("predict [sek]:",end - start)

    #Get only data from "boat" classifications for further processing
    valid3DDetections=[]
    for idx,roiClassification in enumerate(roiClassifications,0):
        if(roiClassification == "boat"):
            cv2.rectangle(imgDbgVis, (validROIs[idx][0],validROIs[idx][1]), (validROIs[idx][2],validROIs[idx][3]), (255, 255, 0), 3)
            boat3dPoint= ROIs3dCenters[idx]
            valid3DDetections.append(boat3dPoint)
    """
    #
    valid3DDetections=[]
    for oneROI3DPt in ROIs3dCenters:
        valid3DDetections.append(oneROI3DPt)

    """
    #visualize 3d Pos of the current boat detections
    for idx,oneDetection in enumerate(valid3DDetections,0):
        frameName = "d" + str(idx)
        tf_broadcaster.sendTransform(oneDetection,
                         (0.0, 0.0, 0.0, 1.0),
                         picStamp,
                         frameName,
                         "map")
    """

    #Track roiDetections over time to validate them
    start = time.time()
    associationTresh = 40#[m]
    tracker.addDetections(valid3DDetections,associationTresh)
    trackedBoats=[]
    trackedBoats=tracker.getTrackings(minTrackingCount=3) # all tracked objects with got trackingcount >= minTrackingCount-times are estimated as boats
    end = time.time()
    print("track boats[sek]:",end - start)

    ####Create debugging visualization
    start = time.time()

    planePose = PoseStamped()
    planePose.pose.position.x = drone3dPosInMap.point.x
    planePose.pose.position.y = drone3dPosInMap.point.y
    planePose.pose.position.z = drone3dPosInMap.point.z
    planePose.header.stamp = picStamp
    planePose.header.frame_id = "map"

    planePath.poses.append(planePose)
    planePathPub.publish(planePath)

    tracks=tracker.getTrackings(minTrackingCount=0)
    #visualize tracks
    for oneTrack in tracks:
        track3DPos = oneTrack.getAveragePos()

        track2DPos= np.uint16(project3DPosToPic(track3DPos,camModel,picStamp))
        crossLen=5
        cv2.line(imgDbgVis, (track2DPos[0],track2DPos[1]-crossLen),(track2DPos[0],track2DPos[1]+crossLen), (0, 255, 255), 2)
        cv2.line(imgDbgVis, (track2DPos[0]-crossLen,track2DPos[1]),(track2DPos[0]+crossLen,track2DPos[1]), (0, 255, 255), 2)
        """
        frameName = "t" + str(oneTrack.id)
        tf_broadcaster.sendTransform(track3DPos,
                         (0.0, 0.0, 0.0, 1.0),
                         picStamp,
                         frameName,
                         "map")
        """
    #visualize 3d Pos of valid tracked objects
    for oneTrack in trackedBoats:
        track3DPos = oneTrack.getAveragePos()
        frameName = "B" + str(oneTrack.id)# + "c" + str(oneTrack.trackingCount)
        tf_broadcaster.sendTransform(track3DPos,
                         (0.0, 0.0, 0.0, 1.0),
                         picStamp,
                         frameName,
                         "map")
        track2DPos= np.uint16(project3DPosToPic(track3DPos,camModel,picStamp))
        crossLen=5
        cv2.line(imgDbgVis, (track2DPos[0],track2DPos[1]-crossLen),(track2DPos[0],track2DPos[1]+crossLen), (0, 255, 0), 2)
        cv2.line(imgDbgVis, (track2DPos[0]-crossLen,track2DPos[1]),(track2DPos[0]+crossLen,track2DPos[1]), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        textpos = (track2DPos[0]+10,track2DPos[1]+10)
        fontScale = 0.7
        fontColor = (0, 255, 0)
        lineType = 2
        cv2.putText(imgDbgVis, str(oneTrack.id),
                    textpos,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    #Visualize Imageframe
    imgHeight, imgWidth, channels = rgb.shape
    # upperleft
    ptul2D = [1,1]
    # upperleft
    ptur2D = [1,imgHeight]
    # upperleft
    ptll2D = [imgWidth,1]
    # upperleft
    ptlr2D = [imgWidth,imgHeight]
    #in image
    color=(255,85,255)
    cv2.rectangle(imgDbgVis, tuple(ptul2D), tuple(ptul2D),color,40)
    cv2.rectangle(imgDbgVis, tuple(ptur2D), tuple(ptur2D),color,40)
    cv2.rectangle(imgDbgVis, tuple(ptll2D), tuple(ptll2D),color,40)
    cv2.rectangle(imgDbgVis, tuple(ptlr2D), tuple(ptlr2D),color,40)
    #in rviz
    #Create Points which mark the edges of the camerapic in the world (RVIZ)
    ptul3D=getPixel3DPosOnWater(ptul2D,drone3dPosInMap,picStamp)
    #upperright
    ptur3D=getPixel3DPosOnWater(ptur2D,drone3dPosInMap,picStamp)
    #lowerleft
    ptll3D=getPixel3DPosOnWater(ptll2D,drone3dPosInMap,picStamp)
    #lowerright
    ptlr3D=getPixel3DPosOnWater(ptlr2D,drone3dPosInMap,picStamp)
    pixelProjection3DPoints = [ptul3D,ptur3D,ptll3D,ptlr3D]
    header = Header()
    header.frame_id = "map"
    header.stamp = picStamp
    image3dProjection = PointCloud2()
    image3dProjection= pc2.create_cloud_xyz32(header,pixelProjection3DPoints)
    picProjPub.publish(image3dProjection)
    end = time.time()
    print("debugvis creation[sek]:",end - start)
    rosPic = cvbridge.cv2_to_imgmsg(imgDbgVis, "rgb8")
    debugPicPub.publish(rosPic)

    """
    #Estimate pixelsize in meter
    pt1=getPixel3DPosOnWater([imgWidth/2,imgHeight/2],drone3dPosInMap,picStamp)
    pt2=getPixel3DPosOnWater([imgWidth/2,imgHeight/2+1],drone3dPosInMap,picStamp)
    dist = np.linalg.norm(pt1-pt2)
    print(dist) #==> 0.45 meter
    """
    endAll = time.time()
    print("================================== sum of all [sek]:",endAll - startAll)

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass


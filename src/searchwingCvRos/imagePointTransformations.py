#!/usr/bin/env python
import cv2
cv2.useOptimized()
import numpy as np

import tf
from geometry_msgs.msg import PointStamped
from searchwingCv import helpers

tf_listener = tf.TransformListener()# must be outside callback

# Function to get the 3D Position of a Pixel on the Water in a Picture by transforming the 2D Pos of the Pixel to a 3D Pos in the World
def getPixel3DPosOnWater(Point2D,Drone3DPosInMap,camModel,stamp):
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
    rayPointOnPlane = helpers.LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
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

#Estimate pixelsize in meter
def getPixelSizeInMeter(imgWidth,imgHeight,drone3dPosInMap, stamp):
    pt1=getPixel3DPosOnWater([imgWidth/2,imgHeight/2],drone3dPosInMap,stamp)
    pt2=getPixel3DPosOnWater([imgWidth/2,imgHeight/2+1],drone3dPosInMap,stamp)
    dist = np.linalg.norm(pt1-pt2)
    print(dist)
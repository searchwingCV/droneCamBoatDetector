#!/usr/bin/env python
####################################################################################
# The central ROS node to receive all datastreams
# It receives images from the camera
# It outputs 3D-position estimates of the detected boats
####################################################################################
import cv2
cv2.useOptimized()
import numpy as np

import sensor_msgs.point_cloud2 as pc2
import rospy
import tf
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge

from searchwingCvRos import imagePointTransformations

tf_broadcaster = tf.TransformBroadcaster()
cvbridge = CvBridge()

class dbgVis():
    def __init__(self):
        self.debugPicPub = rospy.Publisher('detectionsPic', Image, queue_size=1)
        self.picProjPub = rospy.Publisher('picProjection', PointCloud2, queue_size=1)
        self.planePathPub = rospy.Publisher('planePath', Path, queue_size=1)
        self.planePath = Path()
        self.planePath.header.frame_id = "map"
    def setCamIntrinsics(self,i_camIntrinsics):
        self.camModel = i_camIntrinsics;
    # Debugvisualization Publishers

    def createDbgVisImage(self, image, ROIs, tracker, stamp):

        # visualize ROIs
        for oneROI in ROIs:
            cv2.rectangle(image, (oneROI[0], oneROI[1]),
                          (oneROI[2], oneROI[3]), (255, 255, 0), 3)

        # visualize tracks
        allTracks = tracker.getTrackings(minTrackingCount=0)
        trackedBoats = tracker.getTrackings(minTrackingCount=3)
        for oneTrack in allTracks:
            track3DPos = oneTrack.getAveragePos()

            track2DPos = np.uint16(imagePointTransformations.project3DPosToPic(track3DPos, self.camModel, stamp))
            crossLen = 5
            cv2.line(image, (track2DPos[0], track2DPos[1] - crossLen), (track2DPos[0], track2DPos[1] + crossLen),
                     (0, 255, 255), 2)
            cv2.line(image, (track2DPos[0] - crossLen, track2DPos[1]), (track2DPos[0] + crossLen, track2DPos[1]),
                     (0, 255, 255), 2)
            """
            frameName = "t" + str(oneTrack.id)
            tf_broadcaster.sendTransform(track3DPos,
                             (0.0, 0.0, 0.0, 1.0),
                             picStamp,
                             frameName,
                             "map")
            """
        # visualize 3d Pos of valid tracked objects
        for oneTrack in trackedBoats:
            track3DPos = oneTrack.getAveragePos()
            frameName = "B" + str(oneTrack.id)  # + "c" + str(oneTrack.trackingCount)
            tf_broadcaster.sendTransform(track3DPos,
                                         (0.0, 0.0, 0.0, 1.0),
                                         stamp,
                                         frameName,
                                         "map")
            track2DPos = np.uint16(imagePointTransformations.project3DPosToPic(track3DPos, self.camModel, stamp))
            crossLen = 5
            cv2.line(image, (track2DPos[0], track2DPos[1] - crossLen), (track2DPos[0], track2DPos[1] + crossLen),
                     (0, 255, 0), 2)
            cv2.line(image, (track2DPos[0] - crossLen, track2DPos[1]), (track2DPos[0] + crossLen, track2DPos[1]),
                     (0, 255, 0), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            textpos = (track2DPos[0] + 10, track2DPos[1] + 10)
            fontScale = 0.7
            fontColor = (0, 255, 0)
            lineType = 2
            cv2.putText(image, str(oneTrack.id),
                        textpos,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

        # Visualize Imageframe
        imgHeight, imgWidth, channels = image.shape
        # upperleft
        ptul2D = [1, 1]
        # upperleft
        ptur2D = [1, imgHeight]
        # upperleft
        ptll2D = [imgWidth, 1]
        # upperleft
        ptlr2D = [imgWidth, imgHeight]
        # in image
        color = (255, 85, 255)
        cv2.rectangle(image, tuple(ptul2D), tuple(ptul2D), color, 40)
        cv2.rectangle(image, tuple(ptur2D), tuple(ptur2D), color, 40)
        cv2.rectangle(image, tuple(ptll2D), tuple(ptll2D), color, 40)
        cv2.rectangle(image, tuple(ptlr2D), tuple(ptlr2D), color, 40)

        rosPic = cvbridge.cv2_to_imgmsg(image, "rgb8")
        self.debugPicPub.publish(rosPic)


    def createPlanePath(self, image, drone3dPosInMap, stamp):
        ##visualize plane path
        planePose = PoseStamped()
        planePose.pose.position.x = drone3dPosInMap.point.x
        planePose.pose.position.y = drone3dPosInMap.point.y
        planePose.pose.position.z = drone3dPosInMap.point.z
        planePose.header.stamp = stamp
        planePose.header.frame_id = "map"
        self.planePath.poses.append(planePose)
        self.planePathPub.publish(self.planePath)

    def createImageEdges(self, image, drone3dPosInMap, stamp):
        ##Show image edges as 3D Points
        imgHeight, imgWidth, channels = image.shape
        # upperleft
        ptul2D = [1, 1]
        # upperleft
        ptur2D = [1, imgHeight]
        # upperleft
        ptll2D = [imgWidth, 1]
        # upperleft
        ptlr2D = [imgWidth, imgHeight]
        # Create Points which mark the edges of the camerapic in the world (RVIZ)
        ptul3D = imagePointTransformations.getPixel3DPosOnWater(ptul2D, drone3dPosInMap, self.camModel, stamp)
        # upperright
        ptur3D = imagePointTransformations.getPixel3DPosOnWater(ptur2D, drone3dPosInMap, self.camModel, stamp)
        # lowerleft
        ptll3D = imagePointTransformations.getPixel3DPosOnWater(ptll2D, drone3dPosInMap, self.camModel, stamp)
        # lowerright
        ptlr3D = imagePointTransformations.getPixel3DPosOnWater(ptlr2D, drone3dPosInMap, self.camModel, stamp)
        pixelProjection3DPoints = [ptul3D, ptur3D, ptll3D, ptlr3D]
        header = Header()
        header.frame_id = "map"
        header.stamp = stamp
        image3dProjection = PointCloud2()
        image3dProjection = pc2.create_cloud_xyz32(header, pixelProjection3DPoints)
        self.picProjPub.publish(image3dProjection)



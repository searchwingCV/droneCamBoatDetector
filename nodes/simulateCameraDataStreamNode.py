#!/usr/bin/env python
####################################################################################
# The ROS-Node which simulates a camera data stream
# It reads images from a provided video and outputs them through a publisher
####################################################################################
import cv2
cv2.useOptimized()

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#Control the video playback by providing the framenumbers to start and to jump back to start
videoSettings = rospy.get_param('videoReplaySettings')
videopath=videoSettings['path']
startpos =videoSettings['startpos[sek]']*30 # 0*30
rewindpos = videoSettings['rewindpos[sek]']*30 #13*30

class vidReader():
    def __init__(self, path):
        self.path = path
        self.vid = cv2.VideoCapture(self.path)
        if not self.vid.isOpened():
            print "vid not possible to open"
        else:
            print "vid opened!"
    def checkValidFrame(self):
        return self.vid.isOpened()
    def readImgFromVid(self,skipframes):
        skip = skipframes
        while (self.vid.isOpened()):
            ret, frame = self.vid.read()
            skip -= 1
            if skip <0:
                return frame
    def scrollToFrame(self,frame):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES,frame)
    def readFrameNum(self):
        return self.vid.get(cv2.CAP_PROP_POS_FRAMES)

def simulateCamImages():
    # Create Publishers
    pub = rospy.Publisher('chatter', String, queue_size=10)
    picPub = rospy.Publisher('gopro/image_raw',Image,queue_size=1)

    vid = vidReader(videopath)

    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(float(2.0/1)) # 10hz
    cvBridge= CvBridge()

    vid.scrollToFrame(frame=startpos)
    while not rospy.is_shutdown():
        if vid.checkValidFrame == False:
            vid.scrollToFrame(startpos)
        if vid.readFrameNum() > rewindpos:
            vid.scrollToFrame(startpos)
        cvPic = vid.readImgFromVid(skipframes=2)

        rosPic = cvBridge.cv2_to_imgmsg(cvPic, "bgr8")
        rosPic.header.stamp = rospy.Time.now()
        hello_str = "Read new image from video at %s" % rospy.get_time()

        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        picPub.publish(rosPic)
        rate.sleep()

if __name__ == '__main__':
    try:
        simulateCamImages()
    except rospy.ROSInterruptException:
        pass

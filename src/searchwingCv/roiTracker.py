#!/usr/bin/env python
####################################################################################
# Classes to track 3d Positions for further validation checking
# These functions assume that the boats stay nearly at the same position in the world through time
####################################################################################
import cv2
import numpy as np;
from scipy.optimize import linear_sum_assignment
cv2.useOptimized()

# Class which describes one track
initLifetime=3
class Track():
    def __init__(self,pos,id):
        self.lifetime=initLifetime
        #self.pos=np.array([0,0], np.float32)
        self.positionsBuffer = []
        self.id = id
        self.trackingCount=0
        self.positionsBuffer.append(pos)
    def decrLifetime(self):
        self.lifetime=self.lifetime-1
        return self.lifetime
    def incrTrackingCount(self):
        self.trackingCount=self.trackingCount+1
    def setLifetime(self,lifetime):
        self.lifetime=lifetime
    def addNewPos(self,pos):
        self.positionsBuffer.append(pos)
    def getAveragePos(self):
        return np.mean(self.positionsBuffer, axis=0)

# Tracker class
# It uses the hungarian algorithm to determine the best fit a of new set of 3d-detection to a set of old 3d-detections
# The best fit is calculated over a costfunction which uses the euclidean distance of the detections
# If the cost of a fit is below the associationTresh one single tracked is considered as a valid boat
class roiTracker():
    def __init__(self):
        self.tracks = []
        self.id=0

    def getNewId(self):
        newId=self.id
        self.id+=1
        return newId

    def getTrackPoints(self):
        temp=[]
        for oneTrack in self.tracks:
            temp.append(oneTrack.getAveragePos())
        trackPoints=np.asarray(temp)
        return trackPoints

    def addDetections(self,detections,associationTresh):
        if len(detections) == 0:
            return
        if len(self.tracks)==0:
            for oneDet in detections:
                newTrack = Track(oneDet,self.getNewId())
                self.tracks.append(newTrack)
        else:
            trackedPoints=self.getTrackPoints()
            detectionPoints = np.asarray(detections)
            euclidDist = np.linalg.norm(trackedPoints - detectionPoints[:, None], axis=-1)  #calc euclidian distance as cost
            """
            euclidDist = d1-t1 d1-t2 d1-t3 d1-t4...  d=>detected obj t=>tracked obj
                         d2-t1 d2-t2 d2-t3 d2-t4
                         d3-t1 d3-t2 d3-t3 d3-t4
            """
            row_ind, col_ind = linear_sum_assignment(euclidDist) #Calculate best fit (see below) by using the hungarian algorithm
            """
                    => row_ind = sorted => detections
                    => col_ind = assignments => which tracking gets assigned to the detection
            """
            for oneRow_ind in row_ind:
                currentDetection=detectionPoints[oneRow_ind]
                assignedTrackIdx=col_ind[oneRow_ind]
                assignedEuclidDist = euclidDist[oneRow_ind,assignedTrackIdx]
                # If new detection is near old track : assign new detection to old track
                if assignedEuclidDist < associationTresh:
                    self.tracks[assignedTrackIdx].setLifetime(initLifetime)
                    #posold=self.tracks[assignedTrackIdx].getAveragePos()
                    #dist = np.linalg.norm(posold-currentDetection)
                    #print(self.tracks[assignedTrackIdx].id)
                    #print(dist)
                    self.tracks[assignedTrackIdx].addNewPos(currentDetection)
                    self.tracks[assignedTrackIdx].incrTrackingCount()
                # Create new Track
                else:
                    newTrack = Track(currentDetection, self.getNewId())
                    self.tracks.append(newTrack)

        newTracks=[]
        for oneTrack in self.tracks:
            lifeTime=oneTrack.decrLifetime()
            if lifeTime > 0:
                newTracks.append(oneTrack)

        self.tracks=newTracks

    # Get only (valid) trackings, which got detected already minTrackingCount-times
    def getTrackings(self,minTrackingCount):
        trackings=[]
        for oneTrack in self.tracks:
            if oneTrack.trackingCount >= minTrackingCount:
                trackings.append(oneTrack)
        return trackings

#!/usr/bin/env python
####################################################################################
# This file contains all functions to calculate a descriptor for a given image
####################################################################################
import cv2
import numpy as np;
import matplotlib.pyplot as plt

cv2.useOptimized()

dbgImgProcSteps = False

def calcDataStats(data):
    """
    calc basic statistics (mean, stddev) of the given input data
    :param data:
    :return:
    """
    data_flat = data.flatten()
    mean, stddev = cv2.meanStdDev(data_flat)
    """
    skewness=0
    for oneD in data_flat:
        temp=oneD-mean 
        skewness += temp*temp*temp
    skewness = (skewness/len(data_flat))**(1.0/3)
    """
    stats = np.array([mean, stddev]).ravel()
    return stats


def calcPicEntropy(mask, blur):
    """
    Calc Entropy of the binary picture;
    like in "Ship Detection in Optical Remote Sensing Images
    Based on Wavelet Transform and Multi-Level False
    Alarm Identification"
    :param mask: Binary image
    :param blur: Apply blur to picture?
    :return:
    """
    if (blur == True):
        mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=0.5, sigmaY=0.5)
        # plt.imshow(mask,cmap="gray")
        # plt.show()
    maxval = 256  ####### TODO: differentiate for each channels
    binCount = 64

    hist = cv2.calcHist([mask], [0], None, [binCount], [1, maxval])
    hist = cv2.normalize(hist, hist)

    logP = cv2.log(hist)
    entropy = -np.nansum(np.multiply(hist, logP))

    return entropy


def calcNormEigenVals(pts):
    """
    descriptor of the binary image by using eigen values
    :param pts: inpoints
    :return:
    """
    covar, mean = cv2.calcCovarMatrix(pts, mean=None, flags=cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)
    ret, eVal, eVec = cv2.eigen(covar)
    eSum = eVal[0] + eVal[1]
    eVal1 = eVal[0] / eSum
    eVal2 = eVal[1] / eSum
    return eVal1, eVal2


from skimage.feature import greycomatrix, greycoprops
class glcm:
    """
    class to calc the grey level co-occurrence matrices of a picture
    """
    def __init__(self):
        self

    def describe(self, image):
        glcm = greycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        data = []
        data.append(greycoprops(glcm, 'contrast')[0, 0])
        data.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        data.append(greycoprops(glcm, 'homogeneity')[0, 0])
        data.append(greycoprops(glcm, 'ASM')[0, 0])
        data.append(greycoprops(glcm, 'energy')[0, 0])
        data.append(greycoprops(glcm, 'correlation')[0, 0])
        return data

    def getFeatureNames(self):
        FeatNames = ["contrast",
                     "dissimilarity",
                     "homogeneity",
                     "ASM",
                     "energy",
                     "correlation"]

        return FeatNames


class pixelSingleChanStatistics:
    """
    calc basic features like entropy, mean, stddev of a given single image channel
    """
    def __init__(self, debug):
        self.name = ""
        self.debug = debug

    def describe(self, image):
        img_flat = image.flatten()
        entropy = calcPicEntropy(image, blur=False)
        # moments
        stats = calcDataStats(image)
        """
        mean, stddev= cv2.meanStdDev(image)
        skewness=0
        for onePixel in img_flat:
            temp=onePixel-mean 
            skewness += temp*temp*temp
        skewness = (skewness/img_flat.len)**(1.0/3)
        """
        ret = np.hstack((stats, entropy))

        if (self.debug == True):
            for (name, val) in zip(self.getFeatureNames(), ret):
                print(name, val)

        return ret

    def getFeatureNames(self, name):
        FeatNames = [str(name) + "mean",
                     str(name) + "std",
                     str(name) + "entropy"]

        return FeatNames


"""
class fourier:
    dft = cv2.dft(np.float32(img_roi_wave2),flags = cv2.DFT_REAL_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    print(dft.shape)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    plt.imshow(magnitude_spectrum,cmap="gray")
    plt.show()
"""

from skimage.feature import local_binary_pattern
class localBinaryPattern:
    """
    calculate the loacl binary pattern of a given input channel
    these descriptor is good to differenciate between certain img-textures like grass, wall, water, street, etc.
    """
    METHOD = 'uniform'
    """
        * 'uniform': improved rotation invariance with uniform patterns and
        finer quantization of the angular space which is gray scale and
        rotation invariant.
    """

    def __init__(self, radius, points, debug):
        self.radius = radius
        self.points = points
        self.binCount = self.points + 3 + 1
        self.debug = debug

    def describe(self, image):
        lbp = local_binary_pattern(image, self.points, self.radius, self.METHOD)

        if (self.debug):
            n_bins = int(lbp.max() + 1)
            plt.hist(lbp.ravel(), normed=False, bins=n_bins, range=(0, n_bins), facecolor='0.5')
            plt.show()
        hist, bins = np.histogram(lbp.ravel(),
                                  normed=True,
                                  bins=np.arange(0, self.points + 3),
                                  range=(0, self.points + 2))
        # self.binCount=len(bins)
        stats = calcDataStats(hist)
        ret = np.hstack((hist, stats))
        return ret

    def getFeatureNames(self, name):
        FeatNames = []
        for bins in xrange(self.binCount):
            FeatNames.append(str(name) + "LBP" + str(bins))
        return FeatNames


class Histogram:
    """
    calculate the histogram of a given single channel image
    """
    def __init__(self, bins, maxval, debug):
        """
        calc hist
        :param bins: bin count of the hist
        :param maxval: maximum value of the data
        :param debug: if enabled  histogramm is plotted
        """
        self.bins = bins
        self.maxval = maxval
        self.debug = debug

    def describe(self, image, roi_mask):
        # cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
        # cv2.rectangle(image, (region[0], region[1]), (region[2], region[3]), 255, -1)
        # plt.imshow(roi_mask,cmap="gray")
        # plt.show()
        hist = cv2.calcHist([image], [0], roi_mask, [self.bins], [0, self.maxval])
        hist = cv2.normalize(hist, hist).ravel()
        # hist = np.concatenate((hist1,hist2,hist3),axis=0)
        if (self.debug):
            plt.hist(image, normed=True, bins=self.bins, range=(0, self.maxval), facecolor='0.5')
            plt.show()
        stats = calcHistStats(hist)
        ret = np.hstack((hist, stats))
        return ret

    def getFeatureNames(self, name):
        FeatNames = []
        for bins in xrange(self.bins):
            FeatNames.append(str(name) + "Hist" + str(bins))
        FeatNames.append(str(name) + "Hist" + "stddev")
        FeatNames.append(str(name) + "Hist" + "skew")
        return FeatNames


def calcHistStats(data):
    """
    calculate stats which describe the stddev and skewness of the histogramm
    :param data:
    :return:
    """
    data_flat = data.flatten()
    mean, stddev = cv2.meanStdDev(data_flat)
    skewness = 0
    for oneD in data_flat:
        temp = oneD - mean
        skewness += temp * temp * temp
    skewness = (skewness / len(data_flat)) ** (1.0 / 3)
    stats = np.array((stddev, skewness)).ravel()
    return stats


class maskstatistics:
    """
    calculate Eigenvalues 1/2, Ratio of Eigenval1/2, Entropy and Humoments of a given binary image
    check out http://www2.informatik.uni-freiburg.de/~arras/papers/arrasICRA07.pdf for more info
    """
    def __init__(self):
        self

    def describe(self, roi_mask):
        data = []
        pts = np.array(cv2.findNonZero(roi_mask), dtype=np.float32)
        pts = pts.reshape(pts.shape[0], 2)

        eVal1, eVal2 = calcNormEigenVals(pts)
        eRatio = eVal1 / eVal2
        entropy = calcPicEntropy(roi_mask, blur=True)
        Humoments = cv2.HuMoments(cv2.moments(pts)).flatten()
        Humoments = -np.sign(Humoments) * np.log10(np.abs(Humoments))
        ret = np.hstack([eVal1, eVal2, eRatio, entropy, Humoments]).ravel()
        return ret

    # more here: http://www2.informatik.uni-freiburg.de/~arras/papers/arrasICRA07.pdf
    def getFeatureNames(self):
        FeatNames = ["MaskEVal1", "MaskEVal2", "MaskERatio", "MaskEntropy"]
        for i in xrange(7):
            FeatNames.append("MaskHu" + str(i))
        return FeatNames


class Descriptors():
    """
    class to calculate all descriptors
    """
    def __init__(self):
        self.lbpFeat = localBinaryPattern(radius=2, points=16, debug=False)  # 1,8 ; 2,18; 3,24
        self.histFeat = Histogram(16, 255, debug=False)
        self.maskFeat = maskstatistics()
        self.pixstat = pixelSingleChanStatistics(debug=False)
        self.channelNames = ["gray", "hue", "saturation", "value"]
        #self.channelNames = ["gray", "blue", "green", "red", "hue", "saturation", "value"]

    def getDescrNames(self):
        descrNames = []
        for oneChanName in self.channelNames:
            names = np.array((self.pixstat.getFeatureNames(oneChanName),
                              self.histFeat.getFeatureNames(oneChanName),
                              self.lbpFeat.getFeatureNames(oneChanName)))
            names = np.hstack(names)
            descrNames.append(names)
        masknames = np.array(self.maskFeat.getFeatureNames())
        masknames = np.hstack(masknames)
        descrNames.append(masknames)
        ret = np.hstack(descrNames)
        return ret

    def calcDescrROI(self, roi_bgr, roi_mask):
        """
        calculate descriptor of a given color BGR-Image by converting to HSV-channels and describe these channels
        :param roi_bgr: bgr image
        :param roi_mask: single channel binary image which mask the object in the image
        :return: a descriptor of the given image
        """

        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        img_roi_h, img_roi_s, img_roi_v = cv2.split(roi_hsv)
        channels = [roi_gray, img_roi_h, img_roi_s, img_roi_v]
        channelDescriptors = []

        # start = time.time()
        for chan in channels:
            # start = time.time()
            pixstatdescr = self.pixstat.describe(chan)
            # end = time.time()
            # print("pixstat [sek]:",end - start)

            # start = time.time()
            histdescr = self.histFeat.describe(chan, roi_mask)
            # end = time.time()
            # print("histFeat [sek]:",end - start)

            # start = time.time()
            lbpdescr = self.lbpFeat.describe(chan)
            # end = time.time()
            # print("lbpFeat [sek]:",end - start)

            oneChanDescr = np.hstack((pixstatdescr, histdescr, lbpdescr))
            channelDescriptors.append(oneChanDescr)

        channelDescriptors = np.hstack((channelDescriptors))
        # end = time.time()
        # print("channelDescriptors [sek]:",end - start)
        maskdescr = self.maskFeat.describe(roi_mask)
        descriptor = np.hstack((channelDescriptors, maskdescr))
        # descriptor= np.hstack((channelDescriptors)) #skip mask huu description
        return descriptor

from sys import argv, stderr
from os import listdir
from os.path import isdir, isfile, join
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import *

######################################
#			   FUNCTIONS			 #
######################################

#Given the keypoints of an image and the matrix of descriptors with matches return the list of KeyPoints associated
def getKeyPointsFromDescriptorMatch(keypoints, descriptor_matches):
	kp = []
	k = len(descriptor_matches[0])
	for match in descriptor_matches:
		for i in (0,k-1):
			kp.append(keypoints[match[i].queryIdx]) #queryIdx = Index of the KeyPoint with match
	return kp

######################################
#			    MAIN			 	 #
######################################

####### Param specification #######
USE = "Use: <Script Name> <Training Dir> <Test Dir>"
if len(argv) < 3: # < 3 when implemented validation
	printErrorMsg("Param number incorrect\n"+USE)
	exit(1)
tPath = argv[1]
vPath = argv[2]
if not isdir(tPath):
	printErrorMsg("'"+tPath+"'"+" is not a valid directory\n"+USE)
	exit(1)
if (not isdir(vPath)):
	printErrorMsg("'"+vPath+"'"+" is not a valid directory\n"+USE)
	exit(1)

tImages = loadImgs(tPath) #List of Image objects
vImages = loadImgs(vPath)

#####KEYPOINTS CAPTURE#####
orb = cv2.ORB(100,4,1)
desc = [] #Training descriptors

print "\033[93mDetecting KeyPoints of Training Images..\033[0m"
# Find and compute the keypoints with ORB
for img in tImages:
	img.k, img.d = orb.detectAndCompute(img.img, None)
	desc.append(img.d)
printOK()

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
               table_number = 6, # 12
               key_size = 12,     # 20
               multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)
flann.add(desc) #desc => Complete list of descriptors
flann.train()

## EXAMPLE ## This is for visualizing the relationship between images
#img1 = tImages[3]
#img2 = tImages[4]
#matches = flann.match(img1.d,img2.d)
#img3 = drawMatches(img1.img,img1.k,img2.img,img2.k,matches[:10])
#plt.imshow(img3,),plt.show()
#############
print "\033[93mTraining FlannBasedMatcher..\033[0m"
printOK()

print "\033[93mProcessing Testing Images..\033[0m"
for imgV in vImages:
	imgV.k, imgV.d = orb.detectAndCompute(imgV.img, None)
	matchedDescriptors = flann.knnMatch(imgV.d, k=5)
	img = cv2.drawKeypoints(imgV.img, getKeyPointsFromDescriptorMatch(imgV.k, matchedDescriptors))
	#showImage(img) #In case you want to see the keypoints drawn in the picture
printOK()

#### 2. cv2.CascadeClassifier ####

# http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0

print "\033[93mProcessing CascadeClassifier..\033[0m"
classifier = cv2.CascadeClassifier(getInput("XML classifier: "))
for img in vImages:
	car = classifier.detectMultiScale(img.img, 1.3, 2) #Returns an array of rectangles (x,y,w,h)
	for (x,y,w,h) in car:
		cv2.rectangle(img.img,(x,y),(x+w,y+h),(255,0,0),2) #Print the rectangle on the image

	showImage(img.img)

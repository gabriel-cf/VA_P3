from sys import argv, stderr
from os import listdir
from os.path import isdir, isfile, join
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils import drawMatches

######################################
#			   CLASSES				 #
######################################

class Image(object):
	def __init__(self, image, key, descriptor):
		self.img = image
		self.k = key
		self.d = descriptor
		

######################################
#			   FUNCTIONS			 #
######################################

#Prints text to STDERR
def printErrorMsg(text):
	print >> stderr, text
#Given a Path, converts all images to grey scale and returns a list of Image objects
def loadImgs(path):
	return [Image(cv2.imread(join(path, f),0), None, None) for f in listdir(path) if (isfile(join(path, f)) and f.endswith('.jpg'))]
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

tImages = loadImgs(tPath)
vImages = loadImgs(vPath)

#####KEYPOINTS CAPTURE#####
orb = cv2.ORB(100,4,1)
desc = [] #Training descriptors

# Get keypoints and descriptors for each image
for img in tImages:
	#find and compute the keypoints with ORB
	img.k, img.d = orb.detectAndCompute(img.img, None)
	desc.append(img.d)
	
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
               table_number = 6, # 12
               key_size = 12,     # 20
               multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

## EXAMPLE ## This is for seeing the relationship between images
#matches = flann.match(tImages[0].d,tImages[1].d)
#img3 = drawMatches(tImages[0].img,tImages[0].k,tImages[1].img,tImages[1].k,matches[:10])
#plt.imshow(img3,),plt.show()
#############

flann.add(desc)
flann.train()

print "FlannBasedMatcher succesfully trained"

for imgV in vImages:
	imgV.k, imgV.d = orb.detectAndCompute(imgV.img, None)
	matchedDescriptors = flann.knnMatch(imgV.d, k=2)
	img = cv2.drawKeypoints(imgV.img, getKeyPointsFromDescriptorMatch(imgV.k, matchedDescriptors))

	cv2.imshow('Matched Features', img)
	cv2.waitKey(500) #500ms
	cv2.destroyWindow('Matched Features')


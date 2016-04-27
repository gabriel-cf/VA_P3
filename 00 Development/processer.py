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

class KeyPoint(object):
	def __init__(self, key, descriptor):
		self.k = key
		self.d = descriptor
		

######################################
#			   FUNCTIONS			 #
######################################

#Print text to STDERR
def printErrorMsg(text):
	print >> stderr, text
#Given a Path, converts all images to grey scale and returns a list of them
def loadImgs(path):
	return [cv2.imread(join(path, f),0) for f in listdir(path) if (isfile(join(path, f)) and f.endswith('.jpg'))]

######################################
#			    MAIN			 	 #
######################################

####### Param specification #######
debug = 1
USE = "Use: <Script Name> <Training Dir> <Validation Dir>"
if len(argv) < 2: # < 3 when implemented validation
	printErrorMsg("Param number incorrect\n"+USE)
	exit(1)
tPath = argv[1]
#vPath = argv[2]
if not isdir(tPath):
	printErrorMsg("'"+tPath+"'"+" is not a valid directory\n"+USE)
	exit(1)
#if (not isdir(vPath)):
#	printErrorMsg("'"+tPath+"'"+" is not a valid directory\n"+USE)
#	exit(1)

tImages = loadImgs(tPath)
#vImages = loadImgs(vPath)
#####KEYPOINTS CAPTURE#####
orb = cv2.ORB(100,4,1)
kp = []

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
               table_number = 6, # 12
               key_size = 12,     # 20
               multi_probe_level = 1) #2
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)


for img in tImages:
	#print "{0}/{1}".format(x+1,l)
	#find and compute the keypoints with ORB
	k, d = orb.detectAndCompute(img,None)
	kp.append(KeyPoint(k,d))
	flann.add(d) #Adds the descriptor to the matcher

## EXAMPLE ##
matches = flann.match(kp[0].d,kp[1].d)
img3 = drawMatches(tImages[0],kp[0].k,tImages[1],kp[1].k,matches[:10])
plt.imshow(img3,),plt.show()
#############


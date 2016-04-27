from sys import argv, stderr
from os import listdir
from os.path import isdir, isfile, join
import cv2
import numpy as np

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
USE = "Use: <Script Name> <Training Dir> <Validation Dir>"
if len(argv) < 2: # < 3 when implemented validation
	printErrorMsg("Param number incorrect\n"+USE)
	exit(1)
tPath = argv[1]
#vPath = argv[2]
if (not isdir(tPath)):
	printErrorMsg("'"+tPath+"'"+" is not a valid directory\n"+USE)
	exit(1)
#if (not isdir(vPath)):
#	printErrorMsg("'"+tPath+"'"+" is not a valid directory\n"+USE)
#	exit(1)

tImages = loadImgs(tPath)
#vImages = loadImgs(vPath)

#####TESTING#####
print len(tImages)
for x in tImages:
	print x
	cv2.imshow('widow',x)
	cv2.waitKey(500)
	cv2.destroyAllWindows()
#################

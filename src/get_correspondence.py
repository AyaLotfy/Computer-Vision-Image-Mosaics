import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

# Manually identifies corresponding points from two views
def getCorrespondence(imageA, imageB):
	# Display images, select matching points
	fig = plt.figure()
	figA = fig.add_subplot(1,2,1)
	figB = fig.add_subplot(1,2,2)
	# Display the image
	# lower use to flip the image
	figA.imshow(imageA)#,origin='lower')
	figB.imshow(imageB)#,origin='lower')
	plt.axis('image')
	# n = number of points to read
	pts = plt.ginput(n=8, timeout=0)
	print(pts);
	pts = np.reshape(pts, (2, 4, 2))
	return pts

if __name__ == "__main__":
	# Read image file names
#	fileA = raw_input("Please insert the file name for the first image: ")
#	fileB = raw_input("Please insert the file name for the second image: ")
	fileA = 'imageA.jpg';
	fileB = 'imageB.jpg';
	imageA = mpimg.imread(fileA)
	imageB = mpimg.imread(fileB)
	pts = getCorrespondence(imageA, imageB)
	print pts

import numpy as np;
import matplotlib.image as mpimg;
import get_correspondence as getCorresp;

def getStartX(pts, index):
	return pts[0][index][0];

def getStartY(pts, index):
	return pts[0][index][1];

def getEndX(pts, index):
	return pts[1][index][0];

def getEndY(pts, index):
	return pts[1][index][1];

# calcuate H matrix using n matched points from get_correspondence
def calcH(pts):
	# SOLVE b = A h
	n = int(int(pts.size)/4);
	# build b
	b = np.zeros(((2*n),1));
	for i in range(0,n):
		b[2*i] = getEndX(pts,i);
		b[2*i+1] = getEndY(pts,i);
#	print(b.size);
#	print(b);
	# build A
	A = np.zeros(((2*n),8));
	for i in range(0,n):
		A[2*i] = [getStartX(pts,i), getStartY(pts,i),1,0,0,0, -getStartX(pts,i)*getEndX(pts, i), -getStartY(pts,i)*getEndX(pts, i)];
		A[2*i + 1] = [0,0,0,getStartX(pts,i), getStartY(pts,i),1, -getStartX(pts,i)*getEndY(pts, i), -getStartY(pts,i)*getEndY(pts, i)];
	#solve equation
	h = np.linalg.lstsq(A, b)[0];
#	reconstructedB = np.dot(A,h);
#	print('reconstructedB '+ str(reconstructedB));
#	print('shapes for A, h, b, reconstructed ' + str(A.shape) + str(h.shape) + str(b.shape) + str(reconstructedB.shape));
	return h;

#def testH(pts)

def main():
	fileA = 'imageA.jpg';
	fileB = 'imageB.jpg';
	imageA = mpimg.imread(fileA);
	imageB = mpimg.imread(fileB);
	pts = getCorresp.getCorrespondence(imageA, imageB);
	print(pts);
	h = calcH(pts);
	print('H :' + str(h));

main();
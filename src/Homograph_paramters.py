import numpy as np;
import matplotlib.image as mpimg;
import get_correspondence as getCorresp;
#import scipy as sc;

from PIL import Image;

def getStartX(pts, index):
	return pts[0][index][0];

def getStartY(pts, index):
	return pts[0][index][1];

def getEndX(pts, index):
	return pts[1][index][0];

def getEndY(pts, index):
	return pts[1][index][1];

def get_sub_A_matrix(x, y, x_t, y_t):
	subMatrix = np.zeros((2,8));
	subMatrix[0] = np.array([x, y,1,0,0,0, -x*x_t, -y*x_t]);
	subMatrix[1] = np.array([0,0,0,x, y,1, -x*y_t, -y*y_t]);
	return subMatrix;

# Transform point using transform h
def transform(point,h):
	h_2d = np.zeros((3,3));
	for i in range(0,3):
		for j in range(0,3):
			if 3*i+j < 8:
				h_2d[i][j] = h[3*i+j];
	h_2d[2][2] = 1;
#	print('h_2d ' + str(h_2d));
	ret = np.dot(h_2d, point);
	ret[0] = ret[0]/ret[2];
	ret[1] = ret[1]/ret[2];
	ret[2] = 1;
	return ret;

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
		subMatrix = get_sub_A_matrix(getStartX(pts,i) , getStartY(pts,i), getEndX(pts, i), getEndY(pts, i));
		A[2*i] = subMatrix[0];
		A[2*i + 1] = subMatrix[1];
	#solve equation
	h = np.linalg.lstsq(A, b)[0];

	for i in range(0,n):
		iniPoint = np.array([[getStartX(pts, i)], [getStartY(pts, i)], [1]]);
		print(' point ' + str(i) + '\n\t'+ str(np.array([[getEndX(pts, i)], [getEndY(pts, i)], [1]])) + '\n\t' + str(transform(iniPoint, h)));
#	print('shapes for A, h, b, reconstructed ' + str(A.shape) + str(h.shape) + str(b.shape) + str(reconstructedB.shape));
	return h;

def wrapImage(sourceImage , h):
	# here we return new sourceImage with (2*width,2*height), and transform our image into destination .. with interpolating
	height = sourceImage.shape[0];
	width = sourceImage.shape[1];
	print(str(height) + ' ' + str(width) + '\n');
	destinationImage = np.zeros((4*height,4*width,3));
	for i in range(0,height):
		for j in range(0, width):
			# perform multiplication	TODO add width,heigh shift
			mappedPos = transform(np.array([[i],[j],[1]]), h);
			# let's
			if height + mappedPos[0][0] < 4*height and  height + mappedPos[0][0] > 0 and  width + mappedPos[1][0] < 4*width and  width + mappedPos[1][0] > 0:
				destinationImage[height+mappedPos[0][0]][width+mappedPos[1][0]] = sourceImage[i][j];

#	scipy.misc.imsave('outfile.jpg', destinationImage)
	im = Image.fromarray(destinationImage, "RGB");
	im.save("your_file.jpeg");

#	return 0;

def main():
#	rrr = np.array([[1],[2],[3],[4]]);
#	print(rrr.shape);
	fileA = 'imageA.jpg';
	fileB = 'imageB.jpg';
	imageA = mpimg.imread(fileA);
#	print(imageA);
	imageB = mpimg.imread(fileB);
	pts = getCorresp.getCorrespondence(imageA, imageB);
	print(pts);
	h = calcH(pts);
	print('H :' + str(h));
	#wrapping
	wrapImage(imageA, h);
main();
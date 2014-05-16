import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

# Manually identifies corresponding points from two views
def getCorrespondence(imageA, imageB):
	getAuto = int(1);
	num = 20;
	if(getAuto == 0):
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
		pts = plt.ginput(n=num, timeout=0)
		print(pts);
		pts = np.reshape(pts, (2, num/2, 2))
		return pts
	pts = np.array([(290.99673461310101, 168.49247971502086), (351.77211557490591, 132.47743914506236), (381.03433603799715, 139.23025925192962), (455.31535721353646, 161.7396596081536), (471.07193746289317, 148.23401939441919), (511.58885810409652, 175.24529982188801), (507.08697803285168, 359.82238274292507), (403.54373639422113, 319.30546210172179), (390.03809618048672, 341.81486245794582), (290.99673461310101, 359.82238274292507), (5.9867999208391893, 139.23025925192951), (80.267821096378498, 109.96803878883827), (107.27910152384732, 121.22273896695026), (177.05824262814178, 161.73965960815349), (195.06576291312103, 148.23401939441908), (224.32798337621227, 175.2452998218879), (208.57140312685544, 353.06956263605764), (98.275341381357748, 303.54888185236484), (82.518761132000918, 326.05828220858882), (3.7358598852168825, 353.06956263605764)])
	pts = np.reshape(pts, (2, num/2, 2))
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

# ===========================
def wrapImage2(sourceImage , h, refImage):
	# here we return new sourceImage with (2*width,2*height), and transform our image into destination .. with interpolating
	height = sourceImage.shape[0];
	width = sourceImage.shape[1];
	newHeight = 2*height;
	newWidth = 2*width;

#	print(str(height) + ' ' + str(width) + '\n');

	destinationImage = np.zeros((newHeight,newWidth,3), dtype=np.uint8);

	min_mapped_i = int(height-1);
	min_mapped_j = int(width-1);
	max_mapped_i = max_mapped_j = int(0);

	deltaHeight = newHeight/2;
	deltaWidth = newWidth/2;
#	deltaHeight = 0;
#	deltaWidth = 0;

	for i in range(0,height):
		for j in range(0, width):
			# perform multiplication	TODO add width,heigh shift
			mappedPos = transform(np.array([[i],[j],[1]]), h);
			# let's
			mapped_i = int(mappedPos[0][0])
			mapped_j = int(mappedPos[1][0])
			if int(deltaHeight + mapped_i) < newHeight and  int(deltaHeight + mapped_i) > -1 and  int(deltaWidth + mapped_j) < newWidth and  int(deltaWidth + mapped_j) > -1:
				destinationImage[int(deltaHeight+mapped_i)][int(deltaWidth+mapped_j)] = sourceImage[i][j];
				# update bounding box!
				if mapped_i < min_mapped_i:
					min_mapped_i = deltaHeight + mapped_i
				if mapped_i > max_mapped_i:
					max_mapped_i = deltaHeight + mapped_i
				if mapped_j < min_mapped_j:
					min_mapped_j = deltaWidth+mapped_j
				if mapped_j > max_mapped_j:
					max_mapped_j = deltaWidth+mapped_j

	im = Image.fromarray(destinationImage)
	im.save("with_holes.jpg")
	
	# let's Remove black holes FTW
	for i in range(min_mapped_i, max_mapped_i+1):
		for j in range(min_mapped_j, max_mapped_j+1):
			# may be done in more neat way!
			if int(destinationImage[i][j][0]) == 0 and int(destinationImage[i][j][1]) == 0 and int(destinationImage[i][j][2]) == 0:
				# it's black let's get back to it's inverse!
				inv_mapped_pos = InvTransform(np.array([[(i-deltaHeight)],[(j- deltaWidth)],[1]]), h)
				inv_mapped_i = inv_mapped_pos[0][0]
				inv_mapped_j = inv_mapped_pos[1][0]
				if int(inv_mapped_i) < height and  int(inv_mapped_i) > -1 and  int(inv_mapped_j) < width and  int(inv_mapped_j) > -1:
					destinationImage[i][j] = sourceImage[int(inv_mapped_i)][int(inv_mapped_j)]

	im = Image.fromarray(destinationImage)
	im.save("without_holes.jpg")


	# merge original image!
	ref_image_height = refImage.shape[0]
	ref_image_width = refImage.shape[1]
	for i in range(0,ref_image_height):
		for j in range(0, ref_image_width):
			destinationImage[(newHeight/2) + i][(newWidth/2) + j] = refImage[i][j]

	im = Image.fromarray(destinationImage)
	im.save("after_add_ref_image.jpg")

import os
import numpy
from PIL import Image

def get_data_from_dir(dir):
	print("collect data from" + dir)
	class_num = -1
	x = []
	y = []
	for file in os.listdir(dir):
		class_path = os.path.join(dir,file)
		if os.path.isdir(class_path):
			class_num += 1
			print(file + "'s index is", class_num)
			for pic in os.listdir(class_path):
				pic_path = os.path.join(class_path,pic)
				if os.path.isfile(pic_path):
					new_pic = numpy.array(Image.open(pic_path).resize((50,50),Image.ANTIALIAS))
					x.append(new_pic)
					y.append(class_num)
	return (numpy.array(x),numpy.array(y))

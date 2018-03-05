import os
import numpy
from PIL import Image

def get_test_from_dir(dir):
    print("collect data from " + dir)
    x = []
    f = open("imagelist.txt", "r")
    while True:  
    	line = f.readline()
    	if line:
            class_path = os.path.join(dir,line)
            class_path = class_path.split("\n")[0]
            new_pic = numpy.array(Image.open(class_path).resize((256,256),Image.ANTIALIAS))
            x.append(new_pic)
    	else:  
        	break
    f.close()
    return (numpy.array(x))

def get_testpic_from_dir(dir):
    print("collect data from " + dir)
    x = []
    y = []
    for file in os.listdir(dir):
        class_path = os.path.join(dir,file)
        if os.path.isfile(class_path):
            new_pic = numpy.array(Image.open(class_path).resize((256,256),Image.ANTIALIAS))
            x.append(new_pic)
            y.append(file)
    return (numpy.array(x),numpy.array(y))

def get_singlepic_from_dir(dir,name):
    print("collect data from " + dir)
    x = []
    y = []
    for file in os.listdir(dir):
        class_path = os.path.join(dir,file)
        if os.path.isfile(class_path):
            new_pic = numpy.array(Image.open(class_path).resize((256,256),Image.ANTIALIAS))
            if file == name:
                x.append(new_pic)
                y.append(file)
    return (numpy.array(x),numpy.array(y))
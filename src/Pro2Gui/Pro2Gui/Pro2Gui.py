from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys 
import h5py
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from get_data_of_train import get_test_from_dir
from get_data_of_train import get_testpic_from_dir
from get_data_of_train import get_singlepic_from_dir
from sklearn.neighbors import BallTree
import os
import numpy

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# the model so far outputs 3D feature maps (height, width, features)


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

# model.summary()

model.load_weights('five_layers.h5')

# get the image name list
ImageNameList = []
f = open("imagelist.txt", "r")
while True:  
    line = f.readline()
    if line:
        ImageNameList.append(line)
    else:
        break
f.close()

# get all the information from 5613 images
x_test = get_test_from_dir("image")
x_test = x_test.astype('float32')
x_test /= 255
print('x_test shape:', x_test.shape)
print(x_test.shape[0], 'train samples')
pre=model.predict_proba(x_test)

# create ball tree
tree = BallTree(pre)

# get the top10 images from one query
def Image_Retrive(filename):
    HitFile_List = []
    TestImageData,TestImageName = get_singlepic_from_dir("test",filename)# ImageData is the array of image ;ImageName is the array of the names of the picture in the ImageData
    TestImageData = TestImageData.astype('float32')
    TestImageData /= 255
    query_matrix = model.predict_proba(TestImageData)
    distance, inden = tree.query( [query_matrix[0]], k = 11)
    for j in range(0,11):
        HitString = ImageNameList[ inden[0][j] ]
        target_filename = HitString[:-1]
        HitFile_List.append(target_filename)
    
    return(numpy.array(HitFile_List))


class LoginDlg(QDialog):
    def __init__(self, parent=None):
        super(LoginDlg, self).__init__(parent)

        self.target_list = []

        self.setGeometry(1000,100,1000,1000)
        self.usr = QLabel(self)
        self.queryLabel = QLabel(self)
        self.usrLineEdit = QLineEdit(self)
        self.okBtn = QPushButton(self)
        self.cancelBtn = QPushButton(self)

        self.usr.setText("image")
        self.okBtn.setText("search")
        self.cancelBtn.setText("cancel")
        self.queryLabel.setText("query image")

        self.usr.setGeometry(10,10,100,20)
        self.usrLineEdit.setGeometry(120,10,200,20)
        self.okBtn.setGeometry(10,40,100,20)
        self.cancelBtn.setGeometry(120,40,100,20)
        self.queryLabel.setGeometry(120,70,100,20)


        self.okBtn.clicked.connect(self.find_image)
        self.cancelBtn.clicked.connect(self.reject)
        self.setWindowTitle("Image Retrival")

    #def accept(self):
    #    if self.usrLineEdit.text().strip() == "eric" and self.pwdLineEdit.text() == "eric":
    #        super(LoginDlg, self).accept()
    #    else:
    #        QMessageBox.warning(self,
    #                "Warning",
    #                "Wrong!",
    #                QMessageBox.Yes)
    #        self.usrLineEdit.setFocus()



    def find_image(self):
        file_name =  self.usrLineEdit.text().strip()
        self.target_list = Image_Retrive(file_name)

        if len(self.target_list) == 11:

            query_label = QLabel(self)
            query_label.setGeometry(10,100,200,200)
            query_label.setScaledContents(True)

            image1 = QLabel(self)
            image1.setGeometry(0,500,200,200)
            image1.setScaledContents(True)

            image2 = QLabel(self)
            image2.setGeometry(200,500,200,200)
            image2.setScaledContents(True)

            image3 = QLabel(self)
            image3.setGeometry(400,500,200,200)
            image3.setScaledContents(True)

            image4 = QLabel(self)
            image4.setGeometry(600,500,200,200)
            image4.setScaledContents(True)

            image5 = QLabel(self)
            image5.setGeometry(800,500,200,200)
            image5.setScaledContents(True)

            image6 = QLabel(self)
            image6.setGeometry(0,700,200,200)
            image6.setScaledContents(True)

            image7 = QLabel(self)
            image7.setGeometry(200,700,200,200)
            image7.setScaledContents(True)

            image8 = QLabel(self)
            image8.setGeometry(400,700,200,200)
            image8.setScaledContents(True)

            image9 = QLabel(self)
            image9.setGeometry(600,700,200,200)
            image9.setScaledContents(True)

            image10 = QLabel(self)
            image10.setGeometry(800,700,200,200)
            image10.setScaledContents(True)

            pic_query = QImage( "image\\" + self.target_list[0])
            query_label.setPixmap(QPixmap.fromImage(pic_query))
            query_label.show()
                       
            pic1 = QImage( "image\\" + self.target_list[1])
            image1.setPixmap(QPixmap.fromImage(pic1))
            image1.show()

            pic2 = QImage( "image\\" + self.target_list[2])
            image2.setPixmap(QPixmap.fromImage(pic2))
            image2.show()    
                    
            pic3 = QImage( "image\\" + self.target_list[3])
            image3.setPixmap(QPixmap.fromImage(pic3))
            image3.show()          
             
            pic4 = QImage( "image\\" + self.target_list[4])
            image4.setPixmap(QPixmap.fromImage(pic4))
            image4.show()    
                   
            pic5 = QImage( "image\\" + self.target_list[5])
            image5.setPixmap(QPixmap.fromImage(pic5))
            image5.show()     
                   
            pic6 = QImage( "image\\" + self.target_list[6])
            image6.setPixmap(QPixmap.fromImage(pic6))
            image6.show()     
                   
            pic7 = QImage( "image\\" + self.target_list[7])
            image7.setPixmap(QPixmap.fromImage(pic7))
            image7.show()  
                      
            pic8 = QImage( "image\\" + self.target_list[8])
            image8.setPixmap(QPixmap.fromImage(pic8))
            image8.show()
                        
            pic9 = QImage( "image\\" + self.target_list[9])
            image9.setPixmap(QPixmap.fromImage(pic9))
            image9.show()
                        
            pic10 = QImage( "image\\" + self.target_list[10])
            image10.setPixmap(QPixmap.fromImage(pic10))
            image10.show()

            self.target_list = []

        else:
            QMessageBox.warning(self,
                    "Warning",
                    "Wrong!",
                    QMessageBox.Yes)       

app = QApplication(sys.argv)
dlg = LoginDlg()
dlg.show()
dlg.exec_()
app.exit()                                     

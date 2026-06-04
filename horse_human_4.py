import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

form_window = uic.loadUiType('./cat_or_dog.ui')[0]

class Exam(QWidget,form_window):
    def __init__(self):
        super().__init__()
        self.path = None
        self.setupUi(self)
        self.model = load_model('./models/cat_or_dog_classification_0.8532000184059143.h5')
        self.btn_open.clicked.connect(self.button_slot)
        # Designer에서 했던 이름 btn_open


    def button_slot(self):
        self.path = QFileDialog.getOpenFileName(self, 'Open File',
                                    '/home/appletea/workspace_onedevice_2/Python/ML_Data/cat_dog',
                                    'Image Files(*.jpg *.jpeg *.png);;All Files(*)')
        print(self.path)


        pixmap = QPixmap(self.path[0])
        self.lbl_image.setPixmap(pixmap)

        try:
            img = Image.open(self.path[0])
            img = img.convert('RGB')
            img = img.resize((64, 64))
            data = np.array(img)
            data = data/255
            data = data.reshape(1, 64, 64, 3)

            predict_value = self.model.predict(data)
            print(predict_value)
            if predict_value[0] > 0.5:
                self.lbl_result.setText('강아지일 확률' + str((predict_value[0][0]*100)) + '%입니다')
            else:
                self.lbl_result.setText('고양이일 확률' + str((100 - predict_value[0][0]*100)) + '%입니다')
        except:
            print('error')

app = QApplication(sys.argv)
main_window = Exam()
main_window.show()
sys.exit(app.exec_())
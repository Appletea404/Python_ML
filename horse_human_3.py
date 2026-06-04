from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import glob

categories = ['horse', 'human']
model = load_model('./models/horse_or_human_classification_1.0.h5')
model.summary()

img = Image.open('./ML_Data/horse-or-human/test/horse1.jpg')
img.show()
img = img.convert('RGB')
img = img.resize((64,64))
data = np.array(img)
data = data/255
horse_data = data.reshape(1, 64, 64, 3)

print('horse_data', categories[int(model.predict(horse_data)[0][0])])

img = Image.open('./ML_Data/horse-or-human/test/human2.jpg')
img.show()
img = img.convert('RGB')
img = img.resize((64,64))
data = np.array(img)
data = data/255
human_data = data.reshape(1, 64, 64, 3)

print('human_data', categories[int(model.predict(human_data)[0][0])])




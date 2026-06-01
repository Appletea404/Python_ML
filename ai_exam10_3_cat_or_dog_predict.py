from PIL import Image
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import glob

categories = ['cat', 'dog']
model = load_model('./models/cat_or_dog_classification_0.864799976348877.h5')
model.summary()

img = Image.open('./ML_Data/cat_dog/test/cat.01.jpg')
img.show()
img = img.convert('RGB')
img = img.resize((64,64))
data = np.array(img)
data = data/255
cat_data = data.reshape(1, 64, 64, 3)

print('cat_data', categories[int(model.predict(cat_data)[0][0])])

img = Image.open('./ML_Data/cat_dog/test/dog.01.jpg')
img.show()
img = img.convert('RGB')
img = img.resize((64,64))
data = np.array(img)
data = data/255
dog_data = data.reshape(1, 64, 64, 3)

print('dog_data', categories[int(model.predict(dog_data)[0][0])])




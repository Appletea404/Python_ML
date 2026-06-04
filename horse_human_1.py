from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = './ML_Data/horse-or-human/train'
categories = ['human', 'horse']

image_w = 64
image_h = 64

pixel = image_w * image_h

X = []
Y = []

for idx, category in enumerate(categories):
    files = glob.glob(f'{img_dir}/{category}*-*.png')
    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert('RGB')
            data = img.resize((image_w, image_h))
            X.append(np.asarray(data))
            Y.append(idx)
            if i % 50 == 0:
                print(f'[{category}] {i+1}번째: {f}')
        except Exception as e:
            print(category, i, 'error', e)
    print(f'[{category}] 완료: 총 {len([y for y in Y if y == idx])}개 로드')

X = np.array(X, dtype=np.float32)
Y = np.array(Y)
X = X / 255.0

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

np.save('./models/X_train.npy', X_train)
np.save('./models/Y_train.npy', Y_train)
np.save('./models/X_test.npy', X_test)
np.save('./models/Y_test.npy', Y_test)

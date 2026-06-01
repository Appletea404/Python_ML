import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# from ai_exam05_titanic import early_stopping, fit_hist

X_train = np.load('./models/X_train.npy',allow_pickle=True)
Y_train = np.load('./models/Y_train.npy',allow_pickle=True)
X_test = np.load('./models/X_test.npy',allow_pickle=True)
Y_test = np.load('./models/Y_test.npy',allow_pickle=True)
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

model = Sequential()
model.add(Conv2D(32, input_shape=(64,64,3),kernel_size=(3, 3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(32, kernel_size=(3, 3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
early_stopping = EarlyStopping(monitor = 'val_accuracy', patience = 7)
fit_hist = model.fit(X_train, Y_train,batch_size=64,epochs=1000,validation_split=0.15,callbacks=[early_stopping])
score = model.evaluate(X_test, Y_test)
print('Evaluation loss : ',score[0])
print('Evaluation accuracy : ',score[1])
model.save('./models/cat_or_dog_classification_{}.h5'.format(score[1]))

plt.plot(fit_hist.history['loss'],label='loss')
plt.plot(fit_hist.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

plt.plot(fit_hist.history['accuracy'],label='accuracy')
plt.plot(fit_hist.history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.show()



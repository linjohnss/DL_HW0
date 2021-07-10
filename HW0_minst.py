#%%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


# Load training data and testing from mnist
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()
print("\t[Info] train data=", len(x_train_image))
print("\t[Info] test  data=", len(x_test_image))
print("\t[Info] Shape of train data=", x_train_image.shape)
print("\t[Info] Shape of train label=", y_train_label.shape)

def plot_image(image):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='binary')  # cmap (colormap)='binary': display in binary
    plt.show()
plot_image(x_train_image[5])

def plot_multiimages(images, labels, prediction, idx, num=32):
    #fig, ax = plt.subplots(4, 8, figsize=(12, 6))
    if num > 32: num = 32
    for i in range(0, num):
        ax = plt.subplot(4, 8, 1 + i)
        ax.imshow(images[idx], cmap='binary')
        if len(prediction) > 0:
            title = "l={},p={}".format(str(labels[idx]), str(prediction[idx]))
        else:
            title = "l={}".format(str(labels[idx]))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

plot_multiimages(x_train_image, y_train_label, [], 0, 32)
print("\t[Infor] train data=", len(x_train_image))
print("\t[Infor] test data=", len(x_test_image))
print("\t[Infor] xtrain_image: {:s}".format(str(x_train_image.shape)))
print("\t[Infor] xtest_image: {:s}".format(str(x_test_image.shape)))

# Reshape the image from 2D image to 1D image of size 28*28
x_train = x_train_image.reshape(60000, 28 * 28).astype('float32')
x_test = x_test_image.reshape(10000, 28 * 28).astype('float32')

print("\t[After] xtrain: {:s}".format(str(x_train.shape)))  
print("\t[After] xtest: {:s}".format(str(x_test.shape)))

# Normalization [0,255]->[0.0,1.0]  
x_train_norm = x_train / 255
x_test_norm = x_test / 255

# Transfer image label into one-hot-encoding
from tensorflow.python.keras.utils.np_utils import np_utils
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)

# Build a neural network
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense #Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()  # Build Linear Model
model.add(Dense(units=256, input_dim=784, activation='sigmoid'))  # activation=sigmoid, relu, linear, exponential
model.add(Dense(10, activation='softmax'))
model.summary()

# loss=mean_squared_error,mean_absolute_error,categorical_crossentropy..
# optimizer=SGD (Stochastic gradient descent), adam, adagrad
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
train_history = model.fit(x=x_train_norm[0:10000], y=y_TrainOneHot[0:10000], validation_split=0.2, epochs=5,
                          batch_size=20, verbose=2)

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

scores = model.evaluate(x_test_norm, y_TestOneHot)
print()
print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1] * 100.0))
print("\t[Info] Making prediction to x_test_norm")

prediction = model.predict_classes(x_test_norm[0:1000])  # Making prediction and save result to prediction  
print()
print("\t[Info] Show 10 prediction result (From 0):")
print("%s\n" % (prediction[0:10]))
plot_multiimages(x_test_image, y_test_label, prediction, idx=0)

# %%

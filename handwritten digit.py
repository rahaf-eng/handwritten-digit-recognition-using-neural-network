import os 
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend for interactive display
import matplotlib.pyplot as plt
mnist=tf.keras.datasets.mnist   
(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
xtrain=tf.keras.utils.normalize(xtrain,axis=1)
xtest=tf.keras.utils.normalize(xtest,axis=1)
model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128,activation="relu"))

# model.add(tf.keras.layers.Dense(10,activation="softmax"))



# model.compile(optimizer="Adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
# model.fit(xtrain,ytrain,epochs=3)
# model.save("handwritten.keras")
model=tf.keras.models.load_model("handwritten.keras")
loss, accuracy=model.evaluate(xtest,ytest)
print(loss)
print(accuracy)
imagenumber=1
while os.path.isfile(f"D:\\images\\img{imagenumber}.png"):
    try:
        img=cv2.imread(f"D:\\images\\img{imagenumber}.png")[:,:,0]
        img=np.invert(np.array([img]))
        prediction=model.predict(img)
        print("the digit is",np.argmax(prediction))
        plt.figure(figsize=(2,3))    
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("error")
    finally:
        imagenumber +=1
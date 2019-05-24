import tensorflow as tf
import numpy as np
#x=np.load("xtraining.npy")
#y=np.load("ytraining.npy")
x=np.load("xtesting.npy")
y=np.load("ytesting.npy")
xt=x[:int(len(x)*0.8)]
yt=y[:int(len(x)*0.8)]
xv=x[int(len(x)*0.8):]
yv=y[int(len(x)*0.8):]
a,b=np.shape(xt[0])
FILE=r"C:\\Users\\Dell\\Desktop\\holidASY\\vgg_rnn.h5"

"""
(rows,columns,channels)=(240,320,3)
vgg = tf.keras.applications.vgg16.VGG16(input_shape=(rows,columns,channels),weights="imagenet",include_top=False) 
vgg.trainable=False

vgg.summary()
vggmodel=tf.keras.Sequential()
vggmodel.add(vgg)
vggmodel.add(tf.keras.layers.Flatten())
vggmodel.summary()
"""
model=tf.keras.Sequential()
#lstm(units=>dim of output,)
model.add(tf.keras.layers.LSTM(units=50,input_shape=(a,b),return_sequences=True))
model.add(tf.keras.layers.LSTM(units=50,return_sequences=False))
#model.add(tf.keras.layers.LSTM(units=1,input_shape=(100,1),return_sequences=True))

#model.add(tf.keras.layers.Dense())#gets added as a time distributed layer
model.add(tf.keras.layers.Dense(101,activation="softmax"))

6

print("loading")
model=tf.keras.models.load_model(FILE)

print("loaded")
model.compile(loss="sparse_categorical_crossentropy",optimizer="RMSprop",metrics=["accuracy"])
model.summary()
model.fit(xt,yt,epochs=30,batch_size=16,verbose=2,validation_data=(xv,yv))
tf.keras.models.save_model(model=model,filepath=FILE,overwrite=True,include_optimizer=False)
print("saved")
#predict = model.predict(data)
#print(predict)
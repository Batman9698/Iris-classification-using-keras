#Dependencies
import numpy as np #for matrix operations
import pandas as pd #for reading csv files
#for creating the neural network
import keras
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
#for one hot encoding
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
#to split the data into testing and training sets
from sklearn.model_selection import train_test_split

#reading the data
data = pd.read_csv('Iris.csv')
X = data.iloc[:,1:-1].values.astype('float32')
Y = data.iloc[:,-1].values

#convert string labels into numerical labels
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = to_categorical(Y)

#spliting the dataset into training and testing samples
train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size = 0.1,random_state = 1)

#creating the neural network
model = models.Sequential()
model.add(layers.Dense(100,input_dim = 4, activation = 'relu'))#layer 1
model.add(layers.Dense(100,activation = 'relu'))#layer 2
model.add(layers.Dense(100,activation = 'relu'))#layer 3
model.add(layers.Dense(100,activation = 'relu'))#layer 4
model.add(layers.Dense(100,activation = 'relu'))#layer 5

model.add(layers.Dense(3,activation = 'softmax'))#output layer
print(keras.Sequential.summary(model))

model.compile(loss = 'categorical_crossentropy',
				optimizer = 'adam',
				metrics = ['accuracy'])

#training
model.fit(train_x,train_y,epochs = 100,batch_size = 5)

#printing the results
scores = model.evaluate(test_x,test_y)
print("\n%s: %.2f%%"%(model.metrics_names[1],scores[1]*100))


predictions = model.predict_classes(test_x)

prediction = np.argmax(to_categorical(predictions),axis = 1)
#print(prediction)
prediction = encoder.inverse_transform(predictions)

y = np.array(test_y)
y =(np.argmax(y,axis = 1))
y = (encoder.inverse_transform(y))

for i,j in zip(prediction,y):
	print("model prediction: {}, and the species to find is: {}".format(i,j))

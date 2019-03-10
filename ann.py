# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('Churn_Modelling.csv')
features=dataset.iloc[:,3:13].values
labels=dataset.iloc[:,[13]].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x_1=LabelEncoder()
features[:,1]=labelencoder_x_1.fit_transform(features[:,1])
labelencoder_x_2=LabelEncoder()
features[:,2]=labelencoder_x_2.fit_transform(features[:,2])
onehot=OneHotEncoder(categorical_features=[1])
features=onehot.fit_transform(features).toarray()
features=features[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
#sequential is used to initialise the neural network
from keras.layers import Dense
#dense is used to add the layers of the neural network

# Initialising the ANN
classifier=Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(features_train,labels_train,batch_size=10,nb_epoch=100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
pred=classifier.predict(features_test)
pred=(pred>0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,pred)
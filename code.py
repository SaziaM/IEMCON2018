!pip install git+https://github.com/pybrain/pybrain.git

!pip install tensorflow

!pip install pyspark

!pip install keras (not required if its already there)

service_name=input('Streaming Analytics name:')

import getpass
credentials=getpass.getpass('Streaming Analytics credentials:')

%matplotlib inline
%matplotlib notebook

from keras.models import Sequential
from keras.layers import Dense

import numpy as np, math
import matplotlib.pyplot as plt
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# code from Watson Studio insert to code
##################### Code needs to inserted for attaching the data
df_data_1 = pd.read_csv(body)
# load the dataset
dataset = df_data_1.as_matrix()

# fix random seed for reproducibility
np.random.seed(7)

# split into input (X) and output (Y) variables
xvalues = dataset[:,0:-1]
yvalues = dataset[:,-1]

# create model
numCol = len(xvalues[0])
H1 = math.ceil(numCol/2)
H2 = math.ceil(H1/2)
model = Sequential()
model.add(Dense(numCol, input_dim = numCol, activation='relu'))
model.add(Dense(H1, activation='relu'))
model.add(Dense(H2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(xvalues, yvalues, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(xvalues, yvalues)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

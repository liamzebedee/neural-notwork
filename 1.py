import keras
from keras.datasets import mnist
# Sequential model is basically a bunch of NN layers 
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense



# Get the MNIST dataset
# x: byte array of grayscale image data with shape (num_samples, 28, 28).
# y: byte array of digit labels (integers in range 0-9) with shape (num_samples,).
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_samples_training = 60000
num_samples_testing =  10000


# Reshape inputs (x) to a flattened shape
# Why? 
# The naive method we are using in this case, only works on one-dimensional data
# If we were using a more complex construct like a convolutional neural network, this would be useful
x_train = x_train.reshape(num_samples_training, 28*28)
x_test = x_test.reshape(num_samples_testing, 28*28)
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

# We also have to convert the values to floats in the range [0, 1]
# This is because a plurality of reasons:
# 1. If we had the value as a byte [0,255], the weights would be abnormal in comparison to the x
#	 eg. for a pixel of intensity 120, the neuronal activation with weight 0.7
#		 = 0.7 * 120 = 84 ---- places it on far end of sigmoid curve
# 2. GPU's only operate on floats
X_train = x_train.astype('float32')
X_test = x_test.astype('float32')
X_train /= 255 
X_test /= 255


# convert class vectors to binary class matrices
# for use with our cost function (categorical_crossentropy)
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



# Construct the NN
model = Sequential()

# We are going to construct a naive layer that is fully-connected
# It is going to take 784 pixels
# And apply a weight and bias to each one
# And output this to 10 neurons
# Dense implements the operation: output = activation(dot(input, kernel) + bias)
# where kernel is weights created by the layer
input_shape = (784,)
model.add(Dense(num_classes, input_shape=input_shape, activation='softmax'))
model.summary()


# How is it going to train?
# compile: Configures the model for training.

# 1. To train an NN, you need to define something called a cost function
#    The cost function says how poorly the model performs
# 2. You want to minimize the cost, as such to make the model AMAZING!
#    This is done through an optimizer.
#	 The most popular being a stochastic gradient descent optimizer
#	 In SGD we iteratively update our weight parameters in the direction of the gradient of the loss function until we have reached a minimum

# But how can you train the loss function for each neuron when they're all connected in different ways?
# You start at the output neuron (eg. the class - one of numbers 0-9), calculate its cost function
# compared with the predicted cost. 
# You then use SGD to figure out which way is best to go in to MINIMISE the cost.

# But WAIT! There's a problem
# If our output neuron depends on 784 input variables, that's a whole lotta derivatives, Morty!
# We have this crazy invention called backpropagation, also known as reverse-mode differentiation
# It works by computing the partial gradient of each input variable, and then factoring them together to come up with a single equation that we can calc in O(1) time
model.compile(loss='categorical_crossentropy',
              #optimizer=optimizers.SGD(lr=0.5),
              optimizer=optimizers.RMSprop(),
              metrics=['accuracy'])

# Apparently it makes a big difference which optimizer you use!
# Look, change it from RMSProp to SGD



# With what is it going to train?
# Usually we train neural networks in batches
# Why? I'm not completely sure.
# It usually relates to the structure of the network

# We have: training examples (n=900), neural network with 300 neurons (for some silly reason)
# We split the training examples into batches of 300 (to match the NN)
# Each iteration represents running all examples in a batch through the network
# An epoch is when all iterations have run through
# Batch = [0...300]
# Iteration = marks passing a batch through the NN
# Epoch = marks passing all of the training data through the NN
# You might have multiple epochs when you want to keep training but have no more new data


batch_size = 128
epochs = 12

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

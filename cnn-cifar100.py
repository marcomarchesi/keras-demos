# A demo with CIFAR100 dataset
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils, vis_utils
from keras.optimizers import rmsprop

# some parameters
batch_size = 32
num_classes = 100
epochs = 5

# download the dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

# convert to one-hot spots
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# design the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# save model PNG image TODO
# vis_utils.plot_model(model, show_shapes=True, to_file='CIFAR10.png')

# design the optimizer
optimizer = rmsprop(lr=0.0001, decay=1e-6)

# let's train
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=batch_size,
          epochs=epochs, validation_data=(x_test, y_test),
          shuffle=True)







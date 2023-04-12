import tensorflow as tf
from tensorflow import keras

# Load the MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the training and testing data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Convert the training and testing labels to a one-hot encoding
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# Define the neural network architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the network
num_epochs = 10
batch_size = 64

model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size, validation_data=(test_images, test_labels))

# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test set loss {:0.2f} | accuracy {:0.2f}".format(test_loss, test_acc))

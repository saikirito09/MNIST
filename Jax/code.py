import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import nn
from jax.experimental import optimizers
from jax import lax
from jax import tree_map

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mnist.data / 255., mnist.target.astype(int), test_size=0.15)

# Convert the training and testing data to JAX arrays
X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)

# One-hot encode the training and testing labels
y_train = nn.one_hot(y_train, 10)
y_test = nn.one_hot(y_test, 10)

def relu(x):
    return jnp.maximum(0, x)

def init_params(layer_sizes, key):
    keys = random.split(key, len(layer_sizes))
    return [random.normal(keys[i], (layer_sizes[i], layer_sizes[i+1])) * np.sqrt(2/layer_sizes[i]) for i in range(len(layer_sizes)-1)]

def forward(params, x):
    for w in params[:-1]:
        x = relu(jnp.dot(x, w))
    return jnp.dot(x, params[-1])

layer_sizes = [784, 512, 256, 10]
params = init_params(layer_sizes, random.PRNGKey(0))

def loss(params, x, y):
    logits = forward(params, x)
    return -jnp.mean(jnp.sum(y * nn.log_softmax(logits), axis=1))

step_size = 0.001
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

@jit
def update(i, opt_state, x, y):
    params = get_params(opt_state)
    g = grad(loss)(params, x, y)
    return opt_update(i, g, opt_state)

num_epochs = 10
batch_size = 64
num_batches = X_train.shape[0] // batch_size

for epoch in range(num_epochs):
    start_time = time.time()
    for i in range(num_batches):
        batch_x = X_train[i*batch_size:(i+1)*batch_size]
        batch_y = y_train[i*batch_size:(i+1)*batch_size]
        opt_state = update(i, opt_state, batch_x,batch_y)
    epoch_time = time.time() - start_time
    params = get_params(opt_state)
    train_loss = loss(params, X_train, y_train)
    test_loss = loss(params, X_test, y_test)
    train_acc = jnp.mean(jnp.argmax(forward(params, X_train), axis=1) == jnp.argmax(y_train, axis=1))
    test_acc = jnp.mean(jnp.argmax(forward(params, X_test), axis=1) == jnp.argmax(y_test, axis=1))
    print("Epoch {} in {:0.2f} sec".format(epoch+1, epoch_time))
    print("Training set loss {:0.2f} | accuracy {:0.2f}".format(train_loss, train_acc))
    print("Test set loss {:0.2f} | accuracy {:0.2f}".format(test_loss, test_acc))


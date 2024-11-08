import numpy as np

train_images = np.load("../datasets/Kuzushiji-MNIST/kmnist-train-imgs.npz")['arr_0']
train_labels = np.load("../datasets/Kuzushiji-MNIST/kmnist-train-labels.npz")['arr_0']
test_images = np.load("../datasets/Kuzushiji-MNIST/kmnist-test-imgs.npz")['arr_0']
test_labels = np.load("../datasets/Kuzushiji-MNIST/kmnist-test-labels.npz")['arr_0']

print(train_images.shape)
print(train_labels.shape)


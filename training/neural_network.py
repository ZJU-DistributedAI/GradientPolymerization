
import numpy as np

def imageToVector(image):
    return np.reshape(image,[1,784])

def labelToVector(y):
    yLabel = np.zeros([1,10])
    y = int(y)
    yLabel[0][y] = 1
    return yLabel

def normalize_data(ima):
    a_max=np.max(ima)
    a_min=np.min(ima)
    for j in range(ima.shape[1]):
        ima[0][j]=(ima[0][j]-a_min)/(a_max-a_min)
    return ima

def calculate_accuracy(layers, images, labels):

    cnt = 0
    for i in range(len(images)):
        image = imageToVector(images[i])
        image = normalize_data(image)
        activations = simple_forward_step(image,layers)
        output = np.argmax(activations[-1], axis=1)
        if output == labels[i]:
            cnt += 1;
    print(cnt," ",len(labels))
    accuracy = float(cnt) / len(labels)
    return np.round(accuracy,4)


# Forward step that never use homomorphic encryption
def simple_forward_step(input_samples, layers):
    activations = [input_samples]
    for layer in layers:
        activations.append(layer.simple_forward(activations[-1]))
    return activations


# The normal forward step that might use homomorphic encryption
def forward_step(input_samples, layers):
    activations = [input_samples]
    for layer in layers:
        activations.append(layer.forward(activations[-1]))
    return activations


def backward_step(activations, targets, layers, learning_rate):
    parameter = targets
    for index, layer in enumerate(reversed(layers)):
        y = activations.pop()
        x = activations[-1]
        parameter = layer.backward(learning_rate, y, x, parameter)


def perform_simple_training(layers, image, label, settings):
    learning_rate = settings['simple_training_params']['learning_rate']
    activations = forward_step(image, layers)
    backward_step(activations, label, layers, learning_rate)
    return activations



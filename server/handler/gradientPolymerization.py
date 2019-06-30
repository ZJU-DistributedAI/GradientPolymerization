# Import useful libraries.
from sys import argv
import json
from .neural_network import *
from .simple_layer import *
import numpy as np
import time


def imageToVector(image):
    return np.reshape(image, [1, 784])


def labelToVector(y):
    yLabel = np.zeros([1, 10])
    y = int(y)
    yLabel[0][y] = 1
    return yLabel


def normalize_data(ima):
    a_max = np.max(ima)
    a_min = np.min(ima)
    for j in range(ima.shape[1]):
        ima[0][j] = (ima[0][j] - a_min) / (a_max - a_min)
    return ima


def save_json(path, parameter):
    with open(path, 'w') as json_file:
        json.dump(parameter, json_file, ensure_ascii=False)


def load_json(path):
    with open(path, "r") as load_json:
        return json.load(load_json)


def save_txt(path, value):
    value = value + " "
    with open(path, 'a') as f:
        f.write(value)


if __name__ == '__main__':
    settings = {
        # Training method is either 'simple' or 'genetic_algorithm'
        'training_method': 'simple',
        'num_of_images': 600,
        'num_of_eva_images': 10000,
        'batch_size': 10,
        # Parameters used for normal training process
        'simple_training_params': {
            'learning_rate': 0.01
        },
    }

    # read parameter from cli,the parameter is ID of thread
    threadID = argv[1]
    iteration = argv[2]

    # get train data
    numThread = int(threadID)
    numIteration = int(iteration)
    dataPath = "./data/data" + str(numThread + numIteration * 10) + ".json"
    data = load_json(dataPath)
    train_images = data['images']
    train_labels = data['labels']
    print('Starts training mnist')
    # Train.
    i, test_accuracy = 0, []

    # first time get model from server. We use last time's result from second time
    if numIteration == 0:
        primModel = load_json("primModel.json")
    else:
        primModel = load_json("./singleTrainModel/model{}.json".format(threadID))

    model = [
        LinearLayer(primModel['W1'], primModel['b1']),
        ReluLayer(),
        LinearLayer(primModel['W2'], primModel['b2']),
        ReluLayer(),
        LinearLayer(primModel['W3'], primModel['b3']),
        SoftmaxOutputLayer()
    ]

    test_images = np.load('./data/testImages.npy')
    test_labels = np.load('./data/testLabels.npy')
    start = time.time()
    for i in range(settings['num_of_images']):
        image = imageToVector(train_images[i])
        image = normalize_data(image)
        activation = perform_simple_training(model, image, labelToVector(train_labels[i]), settings)
    end = time.time()

    save_txt("./time/Thread{}.txt".format(threadID),str(end - start))

    grad = {'num': settings['num_of_images'],
            'W1': (primModel['W1'] - model[0].w).tolist(),
            'b1': (primModel['b1'] - model[0].b).tolist(),
            'W2': (primModel['W2'] - model[2].w).tolist(),
            'b2': (primModel['b2'] - model[2].b).tolist(),
            'W3': (primModel['W3'] - model[4].w).tolist(),
            'b3': (primModel['b3'] - model[4].b).tolist()}

    gradPath = "./result/grad" + threadID + ".json"
    save_json(gradPath, grad)

    modelParameter = {
            'W1': model[0].w.tolist(),
            'b1': model[0].b.tolist(),
            'W2': model[2].w.tolist(),
            'b2': model[2].b.tolist(),
            'W3': model[4].w.tolist(),
            'b3': model[4].b.tolist()}
    modelPath = "./singleTrainModel/model{}.json".format(threadID)
    save_json(modelPath,modelParameter)

    # test_accuracy_value = calculate_accuracy(model, test_images, test_labels)
    # print(argv[1] + " accuracy: " + str(test_accuracy_value))
    # save_txt("./accuracy/singleAccuracy" + threadID + ".txt", str(test_accuracy_value))

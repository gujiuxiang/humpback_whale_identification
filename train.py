import argparse
import datetime
import os
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras import applications
from keras.layers import Dense, Conv2D, MaxPooling2D, Reshape, Flatten, Input, merge, Lambda, Dropout
from keras.models import Sequential, load_model, save_model
import os
import time
import os
import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)

from tensorflow.python.client import device_lib
from models import *
from data import *
print(device_lib.list_local_devices())
import opts

input_shape = resize_shape
anchor_input = Input(input_shape, name='anchor_input')
positive_category_input = Input(input_shape, name='positive_input')
negative_category_input = Input(input_shape, name='negative_input')

args = opts.parse_opt()

def get_score(id, prediction):
    prediction_list = prediction.split(" ")
    if (prediction_list[0] == id):
        return 1
    if (prediction_list[1] == id):
        return 1.0 / 2
    if (prediction_list[2] == id):
        return 1.0 / 3
    if (prediction_list[3] == id):
        return 1.0 / 4
    if (prediction_list[4] == id):
        return 1.0 / 5
    return 0

def triplet_loss(X, margin=3):
    anchor_embedding, positive_embedding, negative_embedding = X
    positive_distance = K.square(anchor_embedding - positive_embedding)
    negative_distance = K.square(anchor_embedding - negative_embedding)
    positive_distance = K.mean(positive_distance, axis=-1, keepdims=True)
    negative_distance = K.mean(negative_distance, axis=-1, keepdims=True)
    loss = K.maximum(0.0, margin + positive_distance - negative_distance)
    return K.mean(loss)

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def train(training_data, convnet, siamese_net):
    ModelFolder = './weights/'
    ModelSavePath = ModelFolder + 'AllWeights_{epoch:02d}-{loss:.2f}.hdf5'
    CheckPoint = keras.callbacks.ModelCheckpoint(ModelSavePath,
                                                 monitor='loss',
                                                 verbose=0,
                                                 save_best_only=False,
                                                 save_weights_only=False,
                                                 mode='auto',
                                                 period=1)
    CallbacksList = [CheckPoint]

    train_image_list = training_data['image_array']
    train_id_list = training_data['Id']
    train_embedding_list = convnet.predict(np.stack(train_image_list))

    # siamese_net.compile(loss=identity_loss,optimizer=SGD(LEARNING_RATE))
    siamese_net.compile(loss=identity_loss, optimizer=Adam(args.learning_rate))
    training_data_generator = triple_generator(args.batch_size, training_data, resize_shape, augment=True)
    now = time.strftime('%Y.%m.%d.%H.%M.%S')
    loss= []
    for i in range(args.max_epochs):
        history = siamese_net.fit_generator(training_data_generator,
                                            verbose=1,
                                            epochs=1,
                                            steps_per_epoch=100,
                                            workers=20,
                                            use_multiprocessing=True)
        loss.append(history.history['loss'])
        weights_directory = "weights/" + now + '_' + str(i) + "/"
        if not os.path.exists(weights_directory):
            os.makedirs(weights_directory)

        siamese_net.save_weights(weights_directory + "siamese_weights")
        convnet.save_weights(weights_directory + "convnet_weights")
        print("------------------ Epoch {}".format(i))

    plt.plot(loss)
    numpy_loss_history = np.array(loss)
    np.savetxt("loss_history" + now + ".txt", numpy_loss_history, delimiter=",")
    plt.title("Loss function")
    plt.xlabel("epochs")
    plt.show()
    return train_id_list, train_embedding_list, weights_directory

def test(test_data, convnet, siamese_net, train_embedding_list, train_id_list, image_list,id_list):
    test_image_list = test_data['image_array']
    test_id_list = test_data['Id']
    test_embedding_list = convnet.predict(np.stack(test_image_list))
    test_prediction_list = [classify(image_embedding, train_embedding_list, train_id_list) for image_embedding in test_embedding_list]
    test_evaluation = pd.DataFrame({'Id': test_id_list, 'predicted': test_prediction_list}, columns=['Id', 'predicted'])

    weights_folder = 'weights/2018.04.22.07.47.35_45/'

    convnet.load_weights(weights_folder + 'convnet_weights')
    siamese_net.load_weights(weights_folder + 'siamese_weights')
    evaluation_data_generator = triple_generator(args.batch_size, test_data, resize_shape, augment=False)
    evaluation_steps = 10
    #metric_names = siamese_net.metrics_names
    #metric_values = siamese_net.evaluate_generator(evaluation_data_generator, steps=evaluation_steps)
    #print("Metric names", metric_names)
    #print("Metric values", metric_values)

    scores = [get_score(test_id_list[i], test_prediction_list[i]) for i in range(len(test_id_list))]

    print("Test set score:", np.mean(scores))

    embedding_list = convnet.predict(np.stack(image_list))

    from PIL import Image
    sample_sub = pd.read_csv("data/sample_submission.csv")
    submission_file_list = sample_sub['Image']
    submission_image_list = [get_image(f, location='data/test/') for f in submission_file_list]

    submission_embedding_list = convnet.predict(np.stack(submission_image_list))
    for i in range(1, 11):
        threshold = 10 * i
        print("calculating for threshold: " + str(threshold))
        submission_prediction_list = [classify(image_embedding, embedding_list, id_list, threshold=threshold) for
                                      image_embedding in submission_embedding_list]
        submission = pd.DataFrame({'Image': submission_file_list, 'Id': submission_prediction_list},
                                  columns=['Image', 'Id'])
        results_directory = "results/"
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
        submission.to_csv(results_directory + "submission" + str(threshold) + ".csv", index=False)

def test_online():
    args.input_size = 256
    training_data, test_data, image_list, id_list = get_data()

    if args.input_size == 128:
        convnet = build_model_128()
    else:
        convnet = build_model_256()
    weights_folder = 'weights/2018.04.23.03.42.18_196/'
    #model = load_model(weights_folder + 'convnet_weights')
    convnet.load_weights(weights_folder + 'convnet_weights')
    embedding_list = convnet.predict(np.stack(image_list))
    from PIL import Image
    sample_sub = pd.read_csv("data/sample_submission.csv")
    submission_file_list = sample_sub['Image']
    submission_image_list = [get_image(f, location='data/test/') for f in submission_file_list]

    submission_embedding_list = convnet.predict(np.stack(submission_image_list))
    for i in range(1, 11):
        threshold = 10 * i
        print("calculating for threshold: " + str(threshold))
        submission_prediction_list = [classify(image_embedding, embedding_list, id_list, threshold=threshold) for
                                      image_embedding in submission_embedding_list]
        submission = pd.DataFrame({'Image': submission_file_list, 'Id': submission_prediction_list},
                                  columns=['Image', 'Id'])
        results_directory = "results/"
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
        submission.to_csv(results_directory + "submission" + str(threshold) + ".csv", index=False)

def remove_duplicates(li):
    my_set = set()
    filtered = []
    for e in li:
        if e not in my_set:
            filtered.append(e)
            my_set.add(e)
    return filtered


def classify(image_embedding, embedding_list, id_list, num_categories=5, threshold=20):
    image_embedding = np.expand_dims(image_embedding, axis=0)
    stacked_image = np.repeat(image_embedding, len(embedding_list), axis=0)
    square_differences = (stacked_image - embedding_list) ** 2
    scores = np.sum(square_differences, axis=1)
    scores = np.append(scores, [threshold], axis=0)
    id_list = np.append(id_list, ['new_whale'], axis=0)

    sorted_ids = [x for (y, x) in sorted(
        zip(scores, id_list), key=lambda pair: pair[0])]

    return ' '.join(remove_duplicates(sorted_ids)[0:num_categories])


def run():
    if args.input_size == 128:
        convnet = build_model_128()
    else:
        convnet = build_model_256()
    anchor = convnet(anchor_input)
    positive = convnet(positive_category_input)
    negative = convnet(negative_category_input)
    #convnet = make_parallel(convnet, 2)
    loss = merge([anchor, positive, negative], mode=triplet_loss, name='loss', output_shape=(1,))
    siamese_net = Model(input=[anchor_input, positive_category_input, negative_category_input], output=loss)

    if len(args.checkpoint_path)>0:
        print('Init from {}'.format(args.checkpoint_path))
        convnet.load_weights(args.checkpoint_path + '/convnet_weights')
        siamese_net.load_weights(args.checkpoint_path + '/siamese_weights')

    training_data, test_data, image_list, id_list = get_data()

    train_id_list, train_embedding_list = train(training_data, convnet, siamese_net)
    test(test_data, convnet, siamese_net, train_embedding_list, train_id_list, image_list, id_list)

#run()
test_online()


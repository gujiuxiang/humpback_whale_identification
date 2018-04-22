import numpy as np
import pandas as pd
from data_augmentation import random_transform
from random import randint
from collections import Counter
from sklearn.utils import shuffle
from PIL import Image
import opts
args = opts.parse_opt()

if args.input_size == 128:
    resize_shape = (128, 128, 3)
else:
    resize_shape = (256, 256, 3)

def get_image(file, shape=(resize_shape[0],resize_shape[1]), location='data/train/'):
    image = Image.open(location + file)
    image = image.resize(shape)
    image = np.array(image)
    if len(image.shape) == 2:
        image = np.stack([image]*3,axis=2)
    return image

def get_data():
    data = pd.read_csv("data/train.csv")
    data = shuffle(data)
    print('Len of data {}'.format(len(data)))
    resize_shape = (128, 128, 3)
    file_list = data['Image']
    id_list = data['Id']
    image_list = [get_image(f) for f in file_list]
    data['image_array'] = image_list

    print('Create train and test splits ...')
    train_proportion = 0.8
    cutoff_index = int(len(data) * train_proportion)

    training_data = data.iloc[:cutoff_index].reset_index(drop=True)
    test_data = data.iloc[cutoff_index:].reset_index(drop=True)

    print(len(training_data))
    #training_data.keys()
    # training_data['Id']

    training_counts = Counter(training_data['Id'])
    training_data['Id_count'] = training_data.apply(lambda x: training_counts.get(x["Id"]), axis=1)
    test_counts = Counter(test_data['Id'])
    test_data['Id_count'] = test_data.apply(lambda x: test_counts.get(x["Id"]), axis=1)
    print(len(test_data))
    return  training_data, test_data, image_list, id_list

# create generator
def get_triple(len_data, data_images, data_ids, anchor_condition, augment=True):
    anchor_index = np.random.choice(anchor_condition.index[anchor_condition])
    anchor_image = data_images[anchor_index]
    anchor_id = data_ids[anchor_index]

    positive_id = anchor_id
    positive_id_indices = (data_ids == anchor_id)
    positive_id_index = np.random.choice(positive_id_indices.index[positive_id_indices])
    positive_image = data_images[positive_id_index]

    negative_id = anchor_id
    while (anchor_id == negative_id):
        negative_index = randint(0, len_data - 1)
        negative_id = data_ids[negative_index]

    negative_image = data_images[negative_index]

    if augment:
        anchor_image = random_transform(anchor_image)
        positive_image = random_transform(positive_image)
        negative_image = random_transform(negative_image)
    return anchor_image, positive_image, negative_image

def triple_generator(batch_size, data, resize_shape, augment=True):
    len_data = len(data)
    data_images = np.stack(data['image_array'])
    data_ids = data['Id']
    anchor_condition = (data['Id_count'] > 1) & (data_ids != 'new_whale')

    while True:
        anchor_batch = np.zeros((batch_size, resize_shape[0], resize_shape[1], resize_shape[2]))
        positive_image_batch = np.zeros((batch_size, resize_shape[0], resize_shape[1], resize_shape[2]))
        negative_image_batch = np.zeros((batch_size, resize_shape[0], resize_shape[1], resize_shape[2]))
        for i in range(batch_size):
            anchor_batch[i, :, :, :], positive_image_batch[i, :, :, :], negative_image_batch[i, :, :, :] = get_triple(
                len_data, data_images, data_ids, anchor_condition, augment)

        batches = [anchor_batch, positive_image_batch, negative_image_batch]
        yield batches, np.ones(batch_size)

# coding: utf-8

import glob
import numpy as np
import math
import tensorflow as tf

def get_path_files(file_dir):
    """
    Args:
        file_dir: file directory
    Returns:
        list of images
    """
    image_filenames = glob.glob(file_dir + '/*')

    image_list = []
    for file in image_filenames:
        image_list.append(file)

    return image_list

def get_batch(image, image_W, image_H, batch_size, capacity):
    """
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    """

    image = tf.cast(image, tf.string)
    # make an input queue
    input_queue = tf.train.slice_input_producer([image])

    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # image = tf.reshape(image, [image_W, image_H, 3])

    # if you want to test the generated batches of images, you might want to comment the following line.

    # image = tf.image.per_image_standardization(image)

    image_batch = tf.train.batch([image],
                                  batch_size=batch_size,
                                  num_threads=64,
                                  capacity=capacity)

    # image_batch, label_batch = tf.train.shuffle_batch(
    #     [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=3)

    image_batch = tf.cast(image_batch, tf.float32)

    image_batch = tf.divide(image_batch, 255)
    image_batch = tf.multiply(image_batch, 2)
    image_batch = tf.add(image_batch, -1)

    return image_batch
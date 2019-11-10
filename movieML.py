from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf

!pip install -q tensorflow-hub
!pip install -q tensorflow-datasets
import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews", 
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)
from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function 

import argparse
import gzip 
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf


class Mnist():
    class Flags():
        """Set flags"""
        def __init__(self, use_fp16=False, self_test=False):
            self.use_fp16 = use_fp16 
            self.self_test = self_test 

    def __init__(self, 
            source_url= 'https://storage.googleapis.com/cvdf-datasets/mnist/',
            work_directory = 'data',
            image_size = 28,
            num_channels = 1,
            pixel_depth = 255,
            num_labels = 10,
            validation_size = 5000,
            seed = 66478,
            batch_size = 64,
            num_epochs = 2,
            eval_batch_size = 64,
            eval_frequency = 100,
            flags = Flags() 
            ):
        self.source_url = source_url
        self.work_directory = work_directory
        self.image_size = image_size
        self.num_channels = num_channels
        self.pixel_depth = pixel_depth
        self.num_labels = num_labels
        self.validation_size = validation_size
        self.seed = seed
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.eval_batch_size = eval_batch_size
        self.eval_frequency = eval_frequency
        self.flags = flags

    def maybe_download(self, filename):
        """Download the data from Yann's website, unless it's already here."""
        if not tf.gfile.Exists(self.work_directory):
            tf.gfile.MakeDirs(self.work_directory)
        filepath = os.path.join(self.work_directory, filename)
        if not tf.gfile.Exists(filepath):
            filepath, _ = urllib.request.urlretrieve(self.source_url + filename, filepath)
            with tf.gfile.GFile(filepath) as f:
                size = f.size()
            print('Successfully download', filename, size, 'bytes.')
        return filepath

    def data_type(self):
        """Return the type of the activations, weights, and placeholder variables. """
        if self.flags.use_fp16:
            return tf.float16
        else:
            return tf.float32

    def extract_data(self, filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels]
        
        Values are rescaled from [0, 255] down to [-0.5, 0.5]
        """
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            # In order to skip the first 16 bytes
            bytestream.read(16)
            # Read 8x8x60000x1
            buf = bytestream.read(self.image_size * self.image_size * num_images * self.num_channels)
            data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
            data = (data - (self.pixel_depth/2.0)) / self.pixel_depth
            data = data.reshape(num_images, self.image_size, self.image_size, self.num_channels)
            return data

    def extract_labels(self, filename, num_images):
        """Extract the labels into a vector of int64 label IDs"""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1*num_images)
            labels = numpy.frombuffer(buf, dtype = numpy.uint8).astype(numpy.int64)
        return labels

    def fake_data(self, num_images): 
        """Generate a fake dataset that matches the dimensions of MNIST"""
        data = numpy.ndarray(
                shape=(num_images, self.image_size, self.image_size, self.num_channels),
                dtype=numpy.float32)
        label = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
        for image in xrange(num_images):
            label = image % 2
            data[image, :, :, 0] = label - 0.5
            labels[image] = label
        return data, labels

    def error_rate(self, predictions, labels):
        """Return the error rate based on dense predictions and sparse labels"""
        return 100.0 - (
                100.0 * 
                numpy.sum(numpy.argmax(predictions, 1) == labels) /
                predictions.shape[0])

    def main(self):
        if self.flags.self_test:
            print('Running self-test.')
            train_data, train_labels = self.fake_data(256)
            validation_data, validation_labels = self.fake_data(self.eval_batch_size)
            test_data, test_labels = self.fake_data(self.eval_batch_size)
            num_epochs = 1
        else:
            # Get the data
            train_data_filename = self.maybe_download('train-images-idx3-ubyte.gz')
            train_labels_filename = self.maybe_download('train-labels-idx1-ubyte.gz')
            test_data_filename = self.maybe_download('t10k-images-idx3-ubyte.gz')
            test_labels_filename = self.maybe_download('t10k-labels-idx1-ubyte.gz')

            # Extract it into numpy arrays
            train_data = self.extract_data(train_data_filename, 60000)
            train_labels = self.extract_labels(train_labels_filename, 60000)
            test_data = self.extract_data(test_data_filename, 10000)
            test_labels = self.extract_labels(test_labels_filename, 10000)

            # Generate a validation set.
            validation_data = train_data[:self.validation_size, ...]
            validation_labels = train_labels[:self.validation_size]
            train_data = train_data[self.validation_size:, ...]
            train_labels = train_labels[self.validation_size:]
            num_epochs = self.num_epochs
        train_size = train_labels.shape[0]

        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step using the {feed_dict} argument to the Run() call below.
        train_data_node = tf.placeholder(
            self.data_type(),
            shape = (self.batch_size, self.image_size, self.image_size, self.num_channels))

        train_labels_node = tf.placeholder(tf.int64, shape=(self.batch_size,))

        eval_data = tf.placeholder(
            self.data_type(),
            shape = (self.eval_batch_size, self.image_size, self.image_size, self.num_channels))
                
        # The variables below hold all the trainable weights. They are passed an 
        # initial value which will be assigned when we call:
        # {tf.global_variable_initializer().run()}
        conv1_weights = tf.Variable(
                tf.truncated_normal([5, 5, self.num_channels, 32],
                    stddev=0.1,
                    seed=self.seed, dtype = self.data_type()))
        conv1_biases = tf.Variable(tf.zeros([32], dtype = self.data_type()))
        conv2_weights = tf.Variable(tf.truncated_normal(
            [5, 5, 32, 64], stddev=0.1,
            seed=self.seed, dtype=self.data_type()))
        conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype = self.data_type()))
        fc1_weights = tf.Variable(
                tf.truncated_normal([self.image_size // 4 * self.image_size // 4 * 64, 512],
                                    stddev = 0.1,
                                    seed = self.seed,
                                    dtype = self.data_type()))
        fc1_biases = tf.Variable(tf.constant(
            0.1, shape=[512], dtype = self.data_type()))

        fc2_weights = tf.Variable(tf.truncated_normal([512, self.num_labels],
                                                      stddev=0.1,
                                                      seed=self.seed,
                                                      dtype=self.data_type()))
        fc2_biases = tf.Variable(tf.constant(0.1, shape=[self.num_labels], dtype=self.data_type()))

        # We will replicate the model structure for the training subgraph, as well
        # as the evaluation subgraphs, while sharing the training parameters
        def model(data, train=False):
            """ The Model definition. """
            # 2D convolution, with 'SAME' padding (i.e. the output feature map has
            # the same size as the input). Note that {strides} is a 4D array whose
            # shape matches the data layout: [image index, y, x, depth]
            conv = tf.nn.conv2d(data, 
                                conv1_weights,
                                strides=[1,1,1,1],
                                padding='SAME')
            # Bias and rectified linear non-linearity.
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
            # Max pooling. The kernel size spec {ksize} also follows the layout of the data.
            # Here we have a pooling window of 2, and a stride of 2.
            pool = tf.nn.max_pool(
                        relu, 
                        ksize= [1, 2, 2, 1],
                        strides = [1, 2, 2, 1],
                        padding = 'SAME')
            conv = tf.nn.conv2d(
                        pool,
                        conv2_weights,
                        strides=[1,1,1,1],
                        padding = 'SAME')
            relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
            pool = tf.nn.max_pool(
                        relu,
                        ksize=[1,2,2,1],
                        strides=[1,2,2,1],
                        padding='SAME')
            # Reshape the feature map cuboid into a 2D matrix to feed it to the fully connected layers.
            pool_shape = pool.get_shape().as_list()
            reshape = tf.reshape(
                        pool,
                        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            # Fully connected layer. Note that the '+' operation automatically broadcasts the biases.
            hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
            # Add a 50% dropout during training only. Dropout also scales 
            # activations such that no rescaling is needed at evaluation time.
            if train:
                hidden = tf.nn.dropout(hidden, 0.5, seed = self.seed)
            return tf.matmul(hidden, fc2_weights) + fc2_biases

        # Training computation: logits + cross-entropy loss.
        logits = model(train_data_node, True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = train_labels_node, logits = logits))

        # L2 regularization for the fully connected parameters
        regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                        tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

        # Add the regularization term to the loss
        loss += 5e-4 * regularizers

        # Optimizer: set up a variable that's incremented once per batch and 
        # controls the learning rate decay
        batch = tf.Variable(0, dtype=self.data_type())
        # Decay once per epoch, using an exponential schedule starting at 0.01
        learning_rate = tf.train.exponential_decay(
                    0.01, # Base learning rate
                    batch * self.batch_size, # Current index into the dataset.
                    train_size, # Decay step
                    0.95, # Decay rate
                    staircase = True)
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)
        # Predictions for the current training minibatch
        train_prediction = tf.nn.softmax(logits)

        # Predictions for the test and validation, which we'll compute less often
        eval_prediction = tf.nn.softmax(model(eval_data))

        # Small utility function to evaluate a dataset by feeding batches of data to 
        # {eval_data} and pulling the results from {eval_predictions}.
        # Saves memory and enables this to run on smaller GPUs.
        def eval_in_batches(data, sess):
            """ Get all predictions for a dataset by running it in small batches. """
            size = data.shape[0]
            if size < self.eval_batch_size:
                raise ValueError("batch size for evals larger than dataset: %d" % size)
            predictions = numpy.ndarray(shape=(size, self.num_labels), dtype = numpy.float32)
            for begin in xrange(0, size, self.eval_batch_size):
                end = begin + self.eval_batch_size
                if end <= size:
                    predictions[begin:end, :] = sess.run(
                            eval_prediction,
                            feed_dict={eval_data: data[begin:end, ...]})
                else:
                    batch_predictions = sess.run(
                                eval_prediction,
                                feed_dict = {eval_data: data[-self.eval_batch_size:, ...]})
                    predictions[begin:, :] = batch_predictions[begin - size:, :]
            return predictions

        # Create a local session to run the training.
        start_time = time.time()
        with tf.Session() as sess:
            # Run all the initializers to prepare the trainable parameters.
            tf.global_variables_initializer().run()
            print('Initialized!')
            # Loop through training steps.
            for step in xrange(int(num_epochs * train_size) // self.batch_size):
                # Compute the offset of the current minibatch in the data
                # Note that we could use better randomization across epochs.
                offset = (step * self.batch_size) % (train_size - self.batch_size)
                batch_data = train_data[offset:(offset + self.batch_size), ...]
                batch_labels = train_labels[offset:(offset + self.batch_size)]
                # This dictionary maps the batch data (as a numpy array) to the 
                # node in the graph it should be fed to
                feed_dict = {train_data_node: batch_data,
                             train_labels_node: batch_labels}
                # Run the optimizer to update weights
                sess.run(optimizer, feed_dict=feed_dict)
                # print same extra information once reach the evaluation frequency
                if step % self.eval_frequency == 0:
                    # fetch some extra nodes' data
                    l, lr, predictions = sess.run([loss, learning_rate, train_prediction],
                                                 feed_dict=feed_dict)
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('Step %d (epoch %.2f), %.1f ms' %
                            (step, float(step) * self.batch_size / train_size,
                                1000 * elapsed_time / self.eval_frequency))
                    print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    print('Minibatch error: %.1f%%' % self.error_rate(predictions, batch_labels))
                    print('Validation error: %.1f%%' % self.error_rate(
                        eval_in_batches(validation_data, sess), validation_labels))
                    sys.stdout.flush()

            # Finally print the result!
            test_error = self.error_rate(eval_in_batches(test_data, sess), test_labels)
            print('Test error: %.1f%%' % test_error)
            if self.flags.self_test:
                print('test_error', test_error)
                assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (test_error,)

    def run(self):
        self.main()

if __name__ == '__main__':
    mnist = Mnist()
    mnist.run()
    

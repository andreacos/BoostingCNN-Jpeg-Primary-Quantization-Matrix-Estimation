"""
    2019 Department of Information Engineering and Mathematics, University of Siena, Italy.

    Authors:  Andrea Costanzo (andreacos82@gmail.com) and Benedetta Tondi

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details. You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.

    If you are using this software, please cite:

    Y.Niu, B. Tondi, Y.Zhao, M.Barni:
    “Primary Quantization Matrix Estimation of Double Compressed JPEG Images via CNN",
    IEEE Signal Processing Letters, 2019, November
    Available on ArXiv: arXiv preprint:1908.04259  https://arxiv.org/abs/1908.04259

"""

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.activations import softmax as keras_softmax
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy, mean_squared_error
import tensorflow.keras.backend as K
import configuration as cfg


def custom_qf_loss(y_true, y_pred):

    """ Keras custom loss for QF Estimation

    :param y_true: Tensorflow/Theano tensor of predicted labels
    :param y_pred: Tensorflow/Theano tensor of true labels
    :return: Custom loss
    """
    return K.mean(K.pow(2*K.abs(y_pred - y_true), 20) / (1 + K.pow(2*K.abs(y_pred - y_true), 19)))


def custom_softmax_activation(max_vals):

    """
    Keras custom "piece-wise" softmax activation
    :param max_vals: bins for piecewise computation
    :return: custom softmax function
    """
    def custom_softmax(x):

        bins = np.concatenate(([0], np.cumsum(max_vals))).astype(np.int16)

        stack_list = []
        for i in range(0, len(bins)-1):
            stack_list.append(keras_softmax(x[:, int(bins[i]):int(bins[i + 1])]))

        return tf.concat(stack_list, axis=-1)

    return custom_softmax

#@tf.function
def custom_mse_wrapper(max_vals):

    """
    Keras custom "piece-wise" MSE metric
    :param max_vals: bins for piecewise computation
    :return: custom mse function
    """
    def custom_mse(y_true, y_pred):

        bins = np.concatenate(([0], np.cumsum(max_vals))).astype(np.int)

        mse_val = tf.constant(0, dtype=tf.float64)

        for i in range(0, len(bins) - 1):

            y_true_to_coeff = tf.argmax(y_true[:, bins[i]:bins[i+1]], 1) + 1
            y_pred_to_coeff = tf.argmax(y_pred[:, bins[i]:bins[i+1]], 1) + 1

            mse_val = tf.add(mse_val,
                             tf.reduce_sum(tf.square(y_true_to_coeff - y_pred_to_coeff)) / cfg.batch_size)
        return mse_val

    return custom_mse


# Trick to pass parameters to custom losses
def custom_categorical(max_vals):
    def custom_categorical_crossentropy(y_true, y_pred):

        """ Keras custom loss for QF Estimation with the new labeling strategy.
        Now the label is a categorical array with values [0, 1]. The length of the
        array is equal to the sum of values in max_vals, that is the maximum value
        each coefficient can assume. Index == value is set to 1. E.g., for 2 coefficients
        [6, 7] where max_length is [11, 12]:
            y_true = [0 0 0 0 0 1 0 0 0 0 0, 0 0 0 0 0 0 1 0 0 0 0 0]

        Categorical crossentropy is computed on each individual sub-label rather than on full y_true.
        Each "chunk" of loss is added to total loss
        Example:

        loss(y_true[:, 0:11], y_pred[:, 0:11]) + loss(y_true[:, 11:], y_pred[:, 11:])

        :param y_true: Tensorflow/Theano tensor of predicted labels
        :param y_pred: Tensorflow/Theano tensor of true labels
        :return: Custom loss
        """
        bins = np.concatenate(([0], np.cumsum(max_vals))).astype(np.int16)

        loss = categorical_crossentropy(y_true[:, int(bins[0]):int(bins[1])],
                                        y_pred[:, int(bins[0]):int(bins[1])])
        for i in range(1, len(bins) - 1):
            loss = tf.add(loss, categorical_crossentropy(y_true[:, int(bins[i]):int(bins[i + 1])],
                                                         y_pred[:, int(bins[i]):int(bins[i + 1])]))

        return loss

    return custom_categorical_crossentropy

#@tf.function
def custom_two_terms_loss_wrapper(max_vals, c=0.8):
    def custom_two_terms_loss(y_true, y_pred):

        """ Keras custom loss for QF Estimation with the new labeling strategy.
        Now the label is a categorical array with values [0, 1]. The length of the
        array is equal to the sum of values in max_vals, that is the maximum value
        each coefficient can assume. Index == value is set to 1. E.g., for 2 coefficients
        [6, 7] where max_length is [11, 12]:
            y_true = [0 0 0 0 0 1 0 0 0 0 0, 0 0 0 0 0 0 1 0 0 0 0 0]

        Categorical crossentropy is computed on each individual sub-label rather than on full y_true.
        Each "chunk" of loss is added to total loss
        Example:

        loss(y_true[:, 0:11], y_pred[:, 0:11]) + loss(y_true[:, 11:], y_pred[:, 11:])

        :param y_true: Tensorflow/Theano tensor of predicted labels
        :param y_pred: Tensorflow/Theano tensor of true labels
        :return: Custom loss
        """
        bins = np.concatenate(([0], np.cumsum(max_vals))).astype(np.int)
        batch_size = cfg.batch_size

        loss_1 = tf.zeros(shape=(batch_size,), dtype=tf.float32)
        loss_2 = tf.zeros(shape=(batch_size,), dtype=tf.float32)

        # For each "segment" of the predicted label / ground truth
        for i in range(0, len(bins) - 1):

            # First term of the loss
            cce = categorical_crossentropy(y_true[:, bins[i]:bins[i + 1]],
                                           y_pred[:, bins[i]:bins[i + 1]])
            # Add to total
            loss_1 = tf.add(loss_1, cce)

            # Second term of the loss
            # Create a tensor with index position e.g. [0, 1, 2, ... , 15] repeated for all batch
            ind_arr_i = tf.cast(tf.range(int(max_vals[i])), dtype=tf.float32)
            batch_ind_i = tf.repeat(tf.reshape(ind_arr_i, (1, int(max_vals[i]))), batch_size, 0)

            # Find where the ground truth segment is equal to 1.0 (once per segment by construction)
            y_true_i = tf.cast(y_true[:, bins[i]:bins[i+1]], dtype=tf.float32)
            v_i = tf.where(tf.equal(y_true_i, tf.constant(1, dtype=tf.float32)))

            # Reshape the information to be compliant with the batch structure
            v_i = tf.cast(tf.repeat(tf.reshape(v_i[:, 1], (batch_size, 1)), int(max_vals[i]), 1), tf.float32)

            # Batch-wise squared distance from each index to the index where ground truth is 1.0
            delta_sq_i = tf.square(tf.subtract(batch_ind_i, v_i))

            # Batch_wise sum over all distances
            custom_term = tf.reduce_sum(tf.multiply(y_pred[:, bins[i]:bins[i+1]], delta_sq_i), 1)

            loss_2 = tf.add(loss_2, custom_term)

        # Weighted loss
        loss = tf.add(tf.scalar_mul(c, loss_1), tf.scalar_mul((1-c), loss_2))

        # Print / save individual loss terms here. Can't return multiple outputs
        tot_loss = tf.reduce_mean(loss)
        tot_loss_1 = tf.reduce_mean(loss_1)
        tot_loss_2 = tf.reduce_mean(loss_2)
        tf.print('Loss: ', tot_loss, 'First term: ', tot_loss_1, 'Second term: ', tot_loss_2, output_stream=sys.stdout)
        '''log = ' Loss = {:.4f}; term_1 = {:.4f}; term_2 = {:.4f}'\
                  .format(tf.keras.backend.eval(tf.reduce_mean(loss)),
                          tf.keras.backend.eval(tf.reduce_mean(loss_1)),
                          tf.keras.backend.eval(tf.reduce_mean(loss_2)))
        # tf.print(log[0], output_stream=sys.stdout)

        with open('loss_terms.log', 'a') as loss_log:
            loss_log.write(log[0].replace(' Loss = ', '')
                                 .replace(' term_1 = ', '')
                                 .replace(' term_2 = ', '') + '\n')'''

        return loss

    return custom_two_terms_loss


def normalised_mean_squared_error(y_true, y_pred):
    """ Keras custom loss: Normalised Mean Squared Error

    :param y_true: Tensorflow/Theano tensor of predicted labels
    :param y_pred: Tensorflow/Theano tensor of true labels
    :return: Normalised Mean Squared Erro
    """
    return K.mean(K.square(y_pred - y_true) / K.square(y_true), -1)


def change_last_layer_nclasses(model, num_classes, freeze=False):

    """ Changes last (Dense) layer of ContrastNet for transfer learning

    Args:
       model: ContrastNet model.
       num_classes: number of new output classes

    Returns:
       New Keras sequential model.
    """

    for layer in model.layers:
        layer.trainable = not freeze

    # define a new output layer to connect with the last fc layer in vgg
    x = model.layers[-2].output
    new_output_layer = Dense(num_classes, activation='relu', name='predictions')(x)

    # combine the original VGG model with the new output layer
    new_model = Model(inputs=model.input, outputs=new_output_layer)

    return new_model


def contrastNet(in_shape=(64, 64, 3), num_classes=2, nf_base=64, layers_depth=(4, 3)):

    """ Builds the graph for a CNN based on Keras (TensorFlow backend)

    Args:
       in_shape: the shape on the input image (Height x Width x Depth).
       num_classes: number of output classes
       nf_base: number of filters in the first layer
       layers_depth: number of convolutions at each layer

    Returns:
       Keras sequential model.
    """

    model = Sequential()

    # First convolution and Max Pooling (ReLu activation)
    model.add(Conv2D(nf_base, kernel_size=(3, 3), strides=(1, 1), input_shape=in_shape, activation='relu', name='conv1_1'))

    for i in range(0, layers_depth[0]):
        model.add(Conv2D(nf_base+nf_base*(i+1),
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         activation='relu',
                         name='conv1_{}'.format(i+2)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    last_size = nf_base+nf_base*(i+1)

    # Second convolution and Max Pooling (ReLu activation)
    for i in range(0, layers_depth[1]):
        model.add(Conv2D(last_size+nf_base*(i+1),
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         activation='relu',
                         name='conv2_{}'.format(i+2)))

    # Third convolution and Max Pooling (ReLu activation)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    nf = int(model.layers[-1].output_shape[-1]/2)

    model.add(Conv2D(nf, kernel_size=(1, 1), strides=1, name='conv3_1'))

    # Flatten before fully-connected layer(s)
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='relu', name='predictions'))

    return model

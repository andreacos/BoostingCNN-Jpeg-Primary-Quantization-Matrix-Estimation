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

    Boosting CNN-based primary quantization matrix estimation of double JPEG images via a classification-like architecture}, 
    Benedetta Tondi and Andrea Costanzo and Dequ Huang and Bin Li
    ArXiv preprint: https://arxiv.org/abs/2012.00468

"""

import tensorflow as tf
from tensorflow import keras as tfkeras
from time import time
import os
import random
import configuration as cfg
import math
from utils import read_dataset, assign_exp_id, plot_average_epoch_loss, max_min_coefficient
from networks import custom_mse_wrapper, custom_categorical, custom_softmax_activation
from densenet import DenseNet
from batch import next_batch
import numpy as np


if __name__ == '__main__':

    #exp_id_model = assign_exp_id(cfg.q_factors,
    #                  ['-1-terms-loss-from-scratch-coef-{}'.format(cfg.n_epochs, cfg.max_no_Q_coefs)], 'model')
    exp_id_model = 'model_QF1_60-98-s1-1-term-loss-from-86+20+4-ep-{}-coef-{}'.format(cfg.n_epochs, cfg.max_no_Q_coefs)

    # exp_id_results = 'many_qf1_qf2_90-ep-{}-coef-{}'.format(cfg.n_epochs, cfg.max_no_Q_coefs)
    exp_id_results = 'results_QF1_60-98-s1-1-term-loss-from-86+20+4-ep-{}-coef-{}'.format(cfg.n_epochs,cfg.max_no_Q_coefs)

    if not os.path.exists(os.path.join('results', exp_id_results)):
        os.makedirs(os.path.join('results', exp_id_results))

    if not os.path.exists(os.path.join('models', exp_id_model)):
        os.makedirs(os.path.join('models', exp_id_model))

    # -------------- NEW MODEL FROM SCRATCH IS DONE HERE ----------------------------------------
    # Max value for coefficients
    max_coeffs, _ = max_min_coefficient(quality_range=(50, 100),
                                        n_coeffs=cfg.max_no_Q_coefs,
                                        zig_zag_order=cfg.zig_zag_order)

    bins = np.concatenate(([0], np.cumsum(max_coeffs))).astype(np.int16)

    # model, _ = DenseNet(input_shape=(cfg.block_size[0], cfg.block_size[1], 1),
    #                     nb_classes=int(np.sum(max_coeffs)),
    #                     last_activation=custom_softmax_activation(max_vals=max_coeffs))

    from tensorflow.keras.models import load_model
    model = load_model('models/model_QF1=60-95-QF2-90--1-terms-loss-ep-86+20+4-coef-15/model_ep3.h5',
                       custom_objects=({'custom_softmax': custom_softmax_activation(max_coeffs),
                                        'custom_categorical': custom_categorical(max_coeffs),
                                        'custom_mse': custom_mse_wrapper(max_coeffs)}))

    # Display / draw information
    model.summary()
    # plot_model(model, to_file='models/model.png', show_shapes=True)
    opt = tfkeras.optimizers.Adam(lr=cfg.base_lr)

    # Read image paths and labels (i.e. quantisation matrix coefficients) from CSV
    train_images, train_labels, _ = read_dataset(csv_file=os.path.join(cfg.out_train_dir, cfg.training_csv))

    # Determine the number of iterations that complete each epoch (i.e. when the net has seen all the training set)
    max_iterations = int(math.floor(len(train_images) / cfg.batch_size))

    # -------------- CHOOSE THE LOSS FUNCTION  --------------
    model.compile(loss=custom_categorical(max_coeffs),
                  optimizer=opt,
                  metrics=["accuracy", custom_mse_wrapper(max_coeffs)])

    print('The selected training loss is: {}'.format(model.loss.__name__.upper()))
    print('The selected label strategy is weighted: {}'.format(cfg.weighted_label))

    # Start timing
    begin_time = time()

    # Loop epochs
    losses = []

    for ep in range(cfg.n_epochs):

        # Shuffle train data and labels
        perm = list(range(len(train_images)))
        random.shuffle(perm)
        train_images = [train_images[index] for index in perm]
        train_labels = [train_labels[index] for index in perm]

        # Loop training iterations
        for it in range(max_iterations):

            try:
                it_batch, it_labels, it_files = next_batch(batch_size=cfg.batch_size,
                                                           images=train_images,
                                                           labels=train_labels,
                                                           it=it,
                                                           target_size=cfg.block_size,
                                                           max_c_val=max_coeffs)

                # Perform a single iteration
                metrics = model.train_on_batch(it_batch, it_labels)

                # For debug only, check if custom softmax is working
                # pred = model.predict_on_batch(it_batch)
                # for i in range(0, len(bins) - 1):
                #     print(np.round(np.sum(pred[0, bins[i]:bins[i + 1]])))

                losses.append(metrics[0])

                print('Epoch {} Iter {}/{} - Loss: {:3.4f} - MSE: {}'.format(ep, it,
                                                                            max_iterations,
                                                                            metrics[0],
                                                                            metrics[-1]))

            except Exception as ex:
                err_log = open('err.log', 'a+')
                err_log.write('*******ERROR on batch {}! {}******\n'.format(it, str(ex)))
                err_log.close()

            # Save metrics periodically
            if it > 0 and it % cfg.snapshot_frequency == 0:
                print('Epoch {} Iter {}/{} *** Saving metrics ****'.format(ep, it, max_iterations))
                np.save(os.path.join('results', exp_id_results, 'loss.npy'), losses)

        # Print average loss for the epoch that just ended
        out_log = open('{}_output.log'.format(exp_id_model), 'a+')
        avg_loss = plot_average_epoch_loss(losses, ep+1, '', show=False)
        out_log.write('Epoch {} - average loss so far: {}\n'.format(ep, avg_loss))
        out_log.close()

        # Save all final data
        model.save(os.path.join('models', exp_id_model, 'model_ep{}.h5'.format(ep)), True, False)
        np.save(os.path.join('results', exp_id_results, 'loss.npy'), losses)

    elapsed = time() - begin_time
    print('-' * 50)
    print('Training ended after {:5.2f} seconds'.format(elapsed))
    print('-' * 50)

    # Plot iteration loss average
    plot_average_epoch_loss(losses, n_epochs=cfg.n_epochs, exp_identifier=exp_id_results.replace('results', ''))

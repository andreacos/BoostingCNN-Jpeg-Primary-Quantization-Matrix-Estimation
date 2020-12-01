"""
    2020 Department of Information Engineering and Mathematics, University of Siena, Italy.

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

import numpy as np
import cv2
import configuration as cfg
from utils import string2Q, Q2string, coefficient2label, label2coefficient
from tqdm import tqdm
import os


def preprocess_input(im_file, target_size, scale=255.):
    """ Read image and (eventually) scale data
        Arguments:
            im_file     : input image file
            target_size : output size of the image (height, width)
            scale       : pixel scaling value
        Returns:
            The image
    """
    im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    if im.shape != target_size:
        im = cv2.resize(im, target_size)

    return im.astype(np.float32) / scale


def get_label(im_label, max_c_val=None):
    """ Convert image label from a string to an array with number of coefficients
        equal to cfg.max_no_Q_coefs
           Arguments:
               im_label: label string (from csv)
               max_c_val: maximum value for coefficient (coeff to label translation)
           Returns:
               The label array
    """

    if max_c_val is not None:
        coeffs = string2Q(im_label)[:cfg.max_no_Q_coefs]
        labels = coefficient2label(coeffs, max_c_val)
    else:
        labels = string2Q(im_label)[:cfg.max_no_Q_coefs]

    return labels


def weight(coefficients, max_vals, eps):
    """ Create input label and weight it with an eps weight
           Arguments:
               coefficients: label string (from csv)
               max_vals: maximum value for coefficient (coeff to label translation)
               eps: weight for label
           Returns:
               The label array
    """

    w_label = []
    new_label = []
    for idx, cf in enumerate(coefficients):

        N = max_vals[idx]
        i = cf - 1

        w_label = np.zeros((N,))

        w_label[i] = 1 - eps
        delta = 1
        while i - delta >= 0 and i + delta <= N - 1:
            w_label[i - delta] = 0.5 * (1 - eps) * eps ** delta
            w_label[i + delta] = 0.5 * (1 - eps) * eps ** delta
            delta += 1

        # Putting round() after 15 decimals to to machine precision with exps
        while np.round(np.sum(w_label), 15) != 1:

            if i - delta < 0 and i + delta < N - 1:
                w_label[i + delta] = (1 - eps) * eps ** delta

            elif i - delta > 0 and i + delta > N - 1:
                w_label[i - delta] = (1 - eps) * eps ** delta

            elif i - delta < 0 and i + delta == N - 1:
                w_label[i + delta] = eps ** delta

            elif i - delta == 0 and i + delta > N - 1:
                w_label[i - delta] = eps ** delta

            elif i - delta == 0 and i + delta == N - 1:
                w_label[i - delta] = 0.5 * eps ** delta
                w_label[i + delta] = 0.5 * eps ** delta
            else:
                break

            delta += 1

        # Force to 0 all the weights at delta > 1 from the coefficient index. The sum, considering the presence of epsilon,  must still be 1
        #if i == 0:
        #    w_label[i + 2:] = 0
        #    w_label[i + 1] = eps
        #elif i == len(w_label) - 1:
        #    w_label[:i - 1] = 0
        #    w_label[i - 1] = eps
        #else:
        #    w_label[i + 1] = 0.5 * eps
        #    w_label[i - 1] = 0.5 * eps
        #    w_label[i + 2:] = 0
        #    w_label[:i - 1] = 0

        # Accumulate (concatenate) all labels
        if idx == 0:
            new_label = w_label
        else:
            new_label = np.hstack((new_label, w_label))

    return new_label


def get_weighted_label(im_label, max_c_val, epsilon=0.01):
    """ Convert image label from a string to an array with number of coefficients
        equal to cfg.max_no_Q_coefs and weights them (soft-label or perturbation)
           Arguments:
               im_label: label string (from csv)
               max_c_val: maximum value for coefficient (coeff to label translation)
               epsilon: weight for label
           Returns:
               The label array
    """

    coeffs = string2Q(im_label)[:cfg.max_no_Q_coefs]
    labels = weight(coeffs, max_c_val, epsilon)

    return labels


def next_batch(batch_size, images, labels, target_size, it, max_c_val):
    """ Prepare an image batch
        Arguments:
            batch_size    : size of the batch
            images       : list of image paths
            labels       : list of image labels
            target_size   : CNN input size
            it           : training/validation/test iteration. Determine which image to load from the list
            max_c_val    : maximum value for coefficients (used for labels)
        Returns:
            Image batch with shape [batchsize, rows, cols, channels]
            Class labels for each element of the batch
    """

    assert len(images) == len(labels)

    x_batch = []
    y_batch = []
    files = []

    for j in range(batch_size):
        img_name = images[it * batch_size + j]
        img_label = labels[it * batch_size + j]

        img_name = img_name.replace('/media/D/', '/media/amministratore/e257df19-3c9a-4a5e-8ebc-43bc9a6ce05d/')

        x = preprocess_input(img_name, target_size, cfg.scaling_factor_data)

        # Switch between label strategies: perturbed label (cfg.weighted=True) or normal label
        if cfg.weighted_label:
            y = get_weighted_label(img_label, max_c_val, epsilon=cfg.eps_weight)
        else:
            y = get_label(img_label, max_c_val)

        x_batch.append(x)
        y_batch.append(y)
        files.append(img_name)

    return np.expand_dims(np.array(x_batch), -1), np.array(y_batch), files


def evaluate_model(model, images, labels, qfactors, qf_map, target_size,
                   max_samples=None, csv_companion='test_results.csv', coeff_map=None):
    """ Tests a CNN model batch by batch
        Arguments:
            model      : cnn model (Keras)
            images       : list of image paths
            labels       : list of image labels
            qfactors     : list of (QF1, QF2) for images
            qf_map       : table linking each (QF1, QF2) to its Qmatrix coefficients
            target_size   : CNN input size
            max_samples  : number of images in which test is performed. If None, all images
            csv_companion : output file CSV
        Returns:
            Mean loss (average MSE over all the test set)
            Mean accuracy of the model
    """

    if max_samples is None:
        max_samples = len(images)
    else:
        max_samples = min(len(images), max_samples)

    avg_mse = 0.0
    avg_nmse = 0
    acc_exact = 0.0
    mse_x_coef = np.zeros((1, cfg.max_no_Q_coefs))
    acc_x_coef = np.zeros((1, cfg.max_no_Q_coefs))

    with open(csv_companion, 'w') as csv:

        # Write CSV header
        csv.write('Directory;File;QF1;QF2;Acc_exact;Q_coeffs;Pred_Q_coeffs;Abs_dist;MSE;NMSE\n')

        for k in tqdm(range(max_samples), desc='Estimating Quantization Matrix'):

            # Load image & label
            img_name = images[k]
            img_name = img_name.replace('/media/D/', '/media/amministratore/e257df19-3c9a-4a5e-8ebc-43bc9a6ce05d/')

            img_label = labels[k]
            x = preprocess_input(img_name, target_size, cfg.scaling_factor_data)
            true_label = np.array(get_label(img_label))

            # Predict
            predicted_label = model.predict(np.expand_dims(np.expand_dims(x, -1), 0))

            # Some metrics
            total_abs_distance = np.sum(np.abs(true_label - predicted_label))

            # ------------------------------------------------------------------------------------
            # MSE and Normalised MSE (over all coefficients)
            # ------------------------------------------------------------------------------------
            total_mse = np.sum(np.square(true_label - predicted_label)) / cfg.max_no_Q_coefs

            total_nmse = np.sum(np.square(true_label - predicted_label) /
                                np.square(true_label)) / cfg.max_no_Q_coefs

            # ------------------------------------------------------------------------------------
            # MSE and Normalised MSE (for each coefficients)
            # ------------------------------------------------------------------------------------
            avg_mse += total_mse / max_samples
            avg_nmse += total_nmse / max_samples

            mse_x_coef += np.square(true_label - predicted_label)

            # ------------------------------------------------------------------------------------
            # Count how many times each coefficient is predicted EXACTLY
            # ------------------------------------------------------------------------------------
            label_ok = np.round(predicted_label) == true_label
            acc_im = np.count_nonzero(label_ok) / cfg.max_no_Q_coefs
            acc_x_coef += label_ok
            acc_exact += acc_im

            # Dump results to CSV
            csv.write('{};{};{};{};({},{});'
                      '{};{};{};{}\n'.format(os.path.dirname(img_name),
                                             os.path.basename(img_name),
                                             qfactors[k][0],
                                             qfactors[k][1],
                                             acc_im,
                                             Q2string(true_label),
                                             Q2string(predicted_label),
                                             total_abs_distance,
                                             total_mse,
                                             avg_nmse))

    # Return metrics
    return avg_mse, avg_nmse, acc_exact / max_samples, acc_x_coef / max_samples, mse_x_coef / max_samples


def evaluate_model_v20(model, images, labels, qfactors, qf_map, target_size,
                   max_samples=None, coeff_map=None, csv_companion='test_results.csv'):
    """ Tests a CNN model batch by batch
        Arguments:
            model      : cnn model (Keras)
            images       : list of image paths
            labels       : list of image labels
            qfactors     : list of (QF1, QF2) for images
            qf_map       : table linking each (QF1, QF2) to its Qmatrix coefficients
            target_size   : CNN input size
            max_samples  : number of images in which test is performed. If None, all images
            csv_companion : output file CSV
        Returns:
            Mean loss (average MSE over all the test set)
            Mean accuracy of the model
    """

    if max_samples is None:
        max_samples = len(images)
    else:
        max_samples = min(len(images), max_samples)

    avg_mse = 0.0
    avg_nmse = 0
    acc_exact = 0.0
    mse_x_coef = np.zeros((1, cfg.max_no_Q_coefs))
    acc_x_coef = np.zeros((1, cfg.max_no_Q_coefs))

    with open(csv_companion, 'w') as csv:

        # Write CSV header
        csv.write('Directory;File;QF1;QF2;Acc_exact;Q_coeffs;Pred_Q_coeffs;Abs_dist;MSE;NMSE\n')

        for k in tqdm(range(max_samples), desc='Estimating Quantization Matrix'):

            # Load image & label
            img_name = images[k]
            img_name = img_name.replace('/media/D/', '/media/amministratore/e257df19-3c9a-4a5e-8ebc-43bc9a6ce05d/')

            img_label = labels[k]
            x = preprocess_input(img_name, target_size, cfg.scaling_factor_data)
            true_label = np.array(get_label(img_label))

            # Predict
            predicted_label = model.predict(np.expand_dims(np.expand_dims(x, -1), 0))
            predicted_label = label2coefficient(predicted_label.flatten(), max_coefficients=coeff_map)

            # Some metrics
            total_abs_distance = np.sum(np.abs(true_label - predicted_label))

            # ------------------------------------------------------------------------------------
            # MSE and Normalised MSE (over all coefficients)
            # ------------------------------------------------------------------------------------
            total_mse = np.sum(np.square(true_label - predicted_label)) / cfg.max_no_Q_coefs

            total_nmse = np.sum(np.square(true_label - predicted_label) /
                                np.square(true_label)) / cfg.max_no_Q_coefs

            # ------------------------------------------------------------------------------------
            # MSE and Normalised MSE (for each coefficients)
            # ------------------------------------------------------------------------------------
            avg_mse += total_mse / max_samples
            avg_nmse += total_nmse / max_samples

            mse_x_coef += np.square(true_label - predicted_label)

            # ------------------------------------------------------------------------------------
            # Count how many times each coefficient is predicted EXACTLY
            # ------------------------------------------------------------------------------------
            label_ok = np.round(predicted_label) == true_label
            acc_im = np.count_nonzero(label_ok) / cfg.max_no_Q_coefs
            acc_x_coef += label_ok
            acc_exact += acc_im

            # Dump results to CSV
            csv.write('{};{};{};{};({},{});'
                      '{};{};{};{}\n'.format(os.path.dirname(img_name),
                                             os.path.basename(img_name),
                                             qfactors[k][0],
                                             qfactors[k][1],
                                             acc_im,
                                             Q2string(true_label),
                                             Q2string(predicted_label),
                                             total_abs_distance,
                                             total_mse,
                                             avg_nmse))

    # Return metrics
    return avg_mse, avg_nmse, acc_exact / max_samples, acc_x_coef / max_samples, mse_x_coef / max_samples

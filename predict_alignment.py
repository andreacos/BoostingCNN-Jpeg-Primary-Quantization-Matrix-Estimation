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

import configuration as cfg
import os
import numpy as np
from batch import evaluate_model_v20, evaluate_model
from utils import plot_average_accuracy, rearrange_zigzag_array, read_dataset_wfilter_jpeg_grid, qf1_qf2_coefficients_map, max_min_coefficient
from networks import custom_categorical, custom_softmax_activation, custom_two_terms_loss_wrapper, custom_mse_wrapper
from tensorflow.keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


if __name__ == '__main__':

    save_figures = False
    save_data = False
    suppress_csv = True

    # Version 2 has two-terms loss, version 1 has one-term loss
    version = '1.0' # 2.0
    mode = 'aligned' # 'misaligned'

    # Test model
    if version == '2.0':

        used_cnn = 'new_model'
        model_file = 'models/model_QF1_60-98-s1-2-term-loss-from-86+20+4+30+ep-30-coef-15/model_ep29.h5'
        # model_file = 'models/model_QF1_55-98-s1-2-term-loss-from-86+53-ep-30-coef-15/model_ep29.h5'
        output_txt = 'results/results_QF1_60-98-s1-2-term-loss-from-86+20+4+30+ep-30-coef-15/RAISE8K_accuracies_{}_{}.txt'.format(mode.upper(), used_cnn)
        # output_txt = 'results/results_QF1_55-98-s1-2-term-loss-from-86+53-ep-30-coef-15/QF2_80_RAISE8K_MIXED_DATASET_accuracies_{}_{}.txt'.format(mode.upper(), used_cnn)

    elif version == '1.0':

        used_cnn = 'old_model'
        model_file = 'models/model_OriginalPaperModels/DNN90_60LOG.h5'
        # model_file = 'models/model_OriginalPaperModels/DNN80_LC20_From90.h5'
        output_txt = 'results/results_OriginalPaperModels/RAISE8K_QF1<>QF2_accuracies_{}_{}.txt'.format(mode.upper(), used_cnn)
        # output_txt = 'results/results_OriginalPaperModels/QF2_80_RAISE8K_MIXED_DATASET_accuracies_{}_{}.txt'.format(mode.upper(), used_cnn)

    # Output NPY for average accuracy for each coefficient
    per_coeff_acc_file = 'QF2_90_RAISE8K_Average_accuracy_overall_coeff_{}_{}.npy'.format(mode.upper(), used_cnn)
    per_coeff_mse_file = 'QF2_90_RAISE8K_Average_mse_overall_coeff_{}_{}.npy'.format(mode.upper(), used_cnn)

    per_coeff_acc_file_1 = 'QF2_90_RAISE8K_QF1_smaller_QF2_Average_accuracy_overall_coeff_{}_{}.npy'.format(mode.upper(), used_cnn)
    per_coeff_acc_file_2 = 'QF2_90_RAISE8K_QF1_larger_QF2_Average_accuracy_overall_coeff_{}_{}.npy'.format(mode.upper(), used_cnn)

    per_coeff_mse_file_1 = 'QF2_90_RAISE8K_QF1_smaller_QF2_Average_mse_overall_coeff_{}_{}.npy'.format(mode.upper(), used_cnn)
    per_coeff_mse_file_2 = 'QF2_90_RAISE8K_QF1_larger_QF2_Average_mse_overall_coeff_{}_{}.npy'.format(mode.upper(), used_cnn)

    # Data file
    csv_file = os.path.join(cfg.out_test_dir, cfg.test_csv)

    # Load the table linking each pair of JPEG quality factors to the corresponding Q's coefficients
    qf_map = qf1_qf2_coefficients_map(csv_file=csv_file)

    # Max value for coefficients
    max_coeffs, _ = max_min_coefficient(quality_range=(50, 100),
                                        n_coeffs=cfg.max_no_Q_coefs,
                                        zig_zag_order=cfg.zig_zag_order)

    # Load model
    if version == '1.0':
        model = load_model(model_file,
                           custom_objects=({'custom_softmax': custom_softmax_activation(max_coeffs),
                                            'custom_categorical': custom_categorical(max_coeffs)}))
    elif version == '2.0':
        model = load_model(model_file,
                           custom_objects=({'custom_softmax': custom_softmax_activation(max_coeffs),
                                            'custom_two_terms_loss_wrapper': custom_two_terms_loss_wrapper(max_coeffs, cfg.mse_weight),
                                            'custom_mse': custom_mse_wrapper(max_coeffs)}))

    # Read CSV with test dataset for each (QF1, QF2) pair and gris is aligned / misaligned
    arr_accuracy = []
    arr_mse = []
    arr_nmse = []

    avg_acc_matrix = np.zeros((1, cfg.max_no_Q_coefs))
    avg_acc_matrix_1 = np.zeros((1, cfg.max_no_Q_coefs))
    avg_acc_matrix_2 = np.zeros((1, cfg.max_no_Q_coefs))

    avg_mse_matrix = np.zeros((1, cfg.max_no_Q_coefs))
    avg_mse_matrix_1 = np.zeros((1, cfg.max_no_Q_coefs))
    avg_mse_matrix_2 = np.zeros((1, cfg.max_no_Q_coefs))

    tot_pairs_smaller = 0
    tot_pairs_larger = 0
    tot_pairs_with_values = 0
    for qf_pair in cfg.q_factors:

        test_images, test_labels, test_jpeg_pairs = read_dataset_wfilter_jpeg_grid(csv_file=csv_file,
                                                                                   qf_filter=qf_pair,
                                                                                   grid_filter=mode)
        print('Found {} {} images'.format(len(test_images), mode))
        if len(test_images) == 0:
            print('WARNING! NO RECORD FOR {}, {}'.format(qf_pair[0], qf_pair[1]))
            arr_accuracy.append(-1)
            arr_mse.append(-1)
            arr_nmse.append(-1)
            with open(output_txt, 'a') as fp:
                fp.write('-' * 80 + '\n')
                fp.write('QF1 = {} QF2 = {}\n'.format(qf_pair[0], qf_pair[1]))
                fp.write('-' * 80 + '\n')
                fp.write('Test average MSE: {:3.4f}\n'.format(-1))
                fp.write('Test average normalised MSE: {:3.4f}\n'.format(-1))
                fp.write('Test accuracy: {:3.4f}\n'.format(-1))
                fp.write('-'*80 + '\n')
                fp.write('\n')
            continue
        else:
            tot_pairs_with_values+=1

        # Test model performance
        csv_output = 'results/test_results_{}_{}_{}.csv'.format(mode.upper(), qf_pair[0], qf_pair[1])

        print('Version: {}'.format(version))
        if version == '2.0':
            eval_fun = evaluate_model_v20
        else:
            eval_fun = evaluate_model

        avg_mse, avg_nmse, test_accuracy, accuracy_matrix, mse_matrix = \
            eval_fun(model=model,
                     images=test_images,
                     labels=test_labels,
                     qfactors=test_jpeg_pairs,
                     qf_map=qf_map,
                     target_size=cfg.block_size,
                     max_samples=None,
                     coeff_map=max_coeffs,
                     csv_companion=csv_output)

        # If we do not need the CSV files
        if suppress_csv:
            os.remove(csv_output)

        # store data for each pair to average later over all QFs
        arr_accuracy.append(test_accuracy)
        arr_mse.append(avg_mse)
        arr_nmse.append(avg_nmse)

        avg_acc_matrix += accuracy_matrix
        avg_mse_matrix += mse_matrix

        if qf_pair[0] < qf_pair[1]:
            tot_pairs_smaller += 1
            avg_acc_matrix_1 += accuracy_matrix
            avg_mse_matrix_1 += mse_matrix
        else:
            tot_pairs_larger += 1
            avg_acc_matrix_2 += accuracy_matrix
            avg_mse_matrix_2 += mse_matrix


        # Plot average accuracy (over all images) for each coefficient
        # if save_figures:
        #    plot_file_acc = 'results/acc_x_coeff_{}_{}_{}.png'.format(mode.upper(), qf_pair[0], qf_pair[1])
        #    plot_average_accuracy(rearrange_zigzag_array(accuracy_matrix, 8), savefile=plot_file_acc)

        # if save_data:
        #     np.save('accuracy_{}_{}.npy'.format(mode.upper(), qf_pair[1]), arr_accuracy)

        print('JPEG: {} QF1 = {} QF2 = {}'.format(mode.upper(), qf_pair[0], qf_pair[1]))
        print('Test average MSE: {:3.4f}'.format(avg_mse))
        print('Test average normalised MSE: {:3.4f}'.format(avg_nmse))
        print('Test accuracy: {:3.4f}'.format(test_accuracy))
        print('\n')

        with open(output_txt, 'a') as fp:
            fp.write('-' * 80 + '\n')
            fp.write('QF1 = {} QF2 = {}\n'.format(qf_pair[0], qf_pair[1]))
            fp.write('-' * 80 + '\n')
            fp.write('Test average MSE: {:3.4f}\n'.format(avg_mse))
            fp.write('Test average normalised MSE: {:3.4f}\n'.format(avg_nmse))
            fp.write('Test accuracy: {:3.4f}\n'.format(test_accuracy))
            fp.write('-'*80 + '\n')
            fp.write('\n')

    # Average coefficient accuracy / MSE over all (QF1, QF2) pairs
    avg_acc_matrix = avg_acc_matrix / tot_pairs_with_values
    avg_mse_matrix = avg_mse_matrix / tot_pairs_with_values

    avg_acc_matrix_1 = avg_acc_matrix_1 / tot_pairs_smaller
    avg_mse_matrix_1 = avg_mse_matrix_1 / tot_pairs_smaller

    avg_acc_matrix_2 = avg_acc_matrix_2 / tot_pairs_larger
    avg_mse_matrix_2 = avg_mse_matrix_2 / tot_pairs_larger

    np.save(per_coeff_acc_file, avg_acc_matrix)
    np.save(per_coeff_mse_file, avg_mse_matrix)

    np.save(per_coeff_acc_file_1, avg_acc_matrix_1)
    np.save(per_coeff_mse_file_1, avg_mse_matrix_1)

    np.save(per_coeff_acc_file_2, avg_acc_matrix_2)
    np.save(per_coeff_mse_file_2, avg_mse_matrix_2)

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
from utils import plot_average_accuracy, rearrange_zigzag_array, read_dataset_wfilter, qf1_qf2_coefficients_map, max_min_coefficient
from networks import custom_categorical, custom_softmax_activation
from tensorflow.keras.models import load_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


if __name__ == '__main__':

        save_figures = False
        save_data = True
        version = '1.0'

        # Test model
        model_file = 'models/model_OriginalPaperModels/DNN90_60LOG.h5'
        # model_file = 'models/model_OriginalPaperModels/DNN80_LC20_From90.h5'
        # model_file = 'models/model_QF1_55-98_QF2_80_from_ep86qf2_90_53/model_ep52.h5'
        # model_file = 'models/model_QF1_60-98-s1-1-term-loss-from-86+20+4+30-coef-15/model_ep29.h5'
        # model_file = 'models/EPOCHS_0-101_model_60-90_65-90_70-90_75-90_80-90_85-90_90-90_95-90_98-90_ep-40-coef-15/model_ep100.h5'
        # model_file = 'models/EPOCHS_86-100_model_new_perturbed_label/model_ep13.h5'  # That is 87+13=100 epochs
        # model_file = 'models/EPOCHS_86-100_model_new_fully_perturbed_label/model_ep14.h5'
        # model_file = 'models/Model_from_paper/DNN90_60LOG.h5'

        out_txt = 'results/results_OriginalPaperModels/QF2_90_DRESDEN_accuracy_Original_Model80.txt'
        out_acc = 'QF2_90_DRESDEN_Average_overall_accuracy_coeff_old_cnn.npy'
        out_mse = 'QF2_90_DRESDEN_Average_overall_mse_coeff_old_cnn.npy'

        with open(out_txt, 'a') as fp:
            fp.write('Model: {}\n'.format(model_file))

        # Data file
        csv_file = os.path.join(cfg.out_test_dir, cfg.test_csv)

        # Load the table linking each pair of JPEG quality factors to the corresponding Q's coefficients
        qf_map = qf1_qf2_coefficients_map(csv_file=csv_file)

        # Max value for coefficients
        max_coeffs, _ = max_min_coefficient(quality_range=(50, 100),
                                            n_coeffs=cfg.max_no_Q_coefs,
                                            zig_zag_order=cfg.zig_zag_order)

        # Load model
        model = load_model(model_file,
                           custom_objects=({'custom_softmax': custom_softmax_activation(max_coeffs),
                                            'custom_categorical': custom_categorical(max_coeffs)}))

        # Read CSV with test dataset for each (QF1, QF2) pair
        arr_accuracy = []
        arr_mse = []
        arr_nmse = []
        avg_acc_matrix = np.zeros((1, cfg.max_no_Q_coefs))
        avg_mse_matrix = np.zeros((1, cfg.max_no_Q_coefs))

        for qf_pair in cfg.q_factors:

            test_images, test_labels, test_jpeg_pairs = read_dataset_wfilter(csv_file=csv_file,
                                                                             qf_filter=qf_pair)
            if len(test_images) == 0:
                print('WARNING! NO RECORD FOR {}, {}'.format(qf_pair[0], qf_pair[1]))
                break

            # Test model performance
            csv_output = 'results/test_results_{}_{}.csv'.format(qf_pair[0], qf_pair[1])

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

            os.remove(csv_output)

            # store data for each pair to average later over all QFs
            arr_accuracy.append(test_accuracy)
            arr_mse.append(avg_mse)
            arr_nmse.append(avg_nmse)
            avg_acc_matrix += accuracy_matrix
            avg_mse_matrix += mse_matrix

            # Plot average accuracy (over all images) for each coefficient
            if save_figures:
                plot_file_acc = 'results/acc_x_coeff_{}_{}.png'.format(qf_pair[0], qf_pair[1])
                plot_average_accuracy(rearrange_zigzag_array(accuracy_matrix, 8), savefile=plot_file_acc)

            #if save_data:
            #    np.save('accuracy_{}.npy'.format(qf_pair[1]), arr_accuracy)

            print('-' * 80)
            print('QF1 = {} QF2 = {}'.format(qf_pair[0], qf_pair[1]))
            print('-' * 80)
            print('Test average MSE: {:3.4f}'.format(avg_mse))
            print('Test average normalised MSE: {:3.4f}'.format(avg_nmse))
            print('Test accuracy: {:3.4f}'.format(test_accuracy))
            print('-'*80)
            print('\n')

            with open(out_txt, 'a') as fp:
                fp.write('-' * 80 + '\n')
                fp.write('QF1 = {} QF2 = {}\n'.format(qf_pair[0], qf_pair[1]))
                fp.write('-' * 80 + '\n')
                fp.write('Test average MSE: {:3.4f}\n'.format(avg_mse))
                fp.write('Test average normalised MSE: {:3.4f}\n'.format(avg_nmse))
                fp.write('Test accuracy: {:3.4f}\n'.format(test_accuracy))
                fp.write('-'*80 + '\n')
                fp.write('\n')

        # Average coefficient accuracy / MSE over all (QF1, QF2) pairs
        avg_acc_matrix = avg_acc_matrix / len(cfg.q_factors)
        avg_mse_matrix = avg_mse_matrix / len(cfg.q_factors)

        np.save(out_acc, avg_acc_matrix)
        np.save(out_mse, avg_mse_matrix)

        print('-' * 80)
        print('ALL QUALITY FACTORS')
        print('-' * 80)
        print('Test average MSE: {:3.4f}'.format(np.mean(arr_mse)))
        print('Test average normalised MSE: {:3.4f}'.format(np.mean(arr_nmse)))
        print('Test accuracy: {:3.4f}'.format(np.mean(arr_accuracy)))
        print('-' * 80)

        #with open(
        #        'results/results_OriginalPaperModels/PHOTOSHOP_accuracy_Original_Model90.txt', 'a') as fp:
        #    fp.write('-' * 80 + '\n')
        #    fp.write('ALL QUALITY FACTORS\n')
        #    fp.write('-' * 80 + '\n')
        #    fp.write('Test average MSE: {:3.4f}\n'.format(np.mean(arr_mse)))
        #    fp.write('Test average normalised MSE: {:3.4f}\n'.format(np.mean(arr_nmse)))
        #    fp.write('Test accuracy: {:3.4f}\n'.format(np.mean(arr_accuracy)))
        #    fp.write('-'*80 + '\n')
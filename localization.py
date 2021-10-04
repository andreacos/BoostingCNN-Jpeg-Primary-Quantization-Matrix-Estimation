import tensorflow as tf
import utils
import os
from networks import custom_two_terms_loss_wrapper, custom_softmax_activation, custom_mse_wrapper
import numpy as np
import cv2
from skimage.util import view_as_windows
from utils import max_min_coefficient
from matplotlib import pyplot as plt


def window_decision(img_data, model, max_coeffs, win_size=(64, 64), stride=1):
    """Divide input image into blocks then classify each block with a trained CNN model. Block scores are aggregated
       by means of average soft decision **on probability of being contrast enhanced**.

    Args:
       img_data: image matrix.
       model: trained CNN model (Keras)
       win_size: (height, width) of the image blocks. Default: (64, 64)
        stride: window stride
    Returns:
       Score
    """

    # Divide image into overlapping blocks
    image_view = view_as_windows(img_data, [win_size[0], win_size[1], 1], stride)

    h, w, c, _, _, _ = image_view.shape

    # Reshape the image view so that it is a stack of size N_blocks. Each element is a color patch BxBx3
    slices = image_view.reshape(h * w * c, win_size[0], win_size[1], 1)

    # Test the stack
    prediction = model.predict(slices, verbose=0)

    # Decode labels
    decoded_labels = np.zeros((prediction.shape[0], 15))
    for i, p in enumerate(prediction):
        decoded_labels[i, :] = utils.label2coefficient(p.flatten(), max_coefficients=max_coeffs)

    # Reshape decoded labels to the final map
    decoded_map = np.reshape(decoded_labels, (h, w, 15))
    decoded_map = cv2.resize(decoded_map, (img_data.shape[0], img_data.shape[1]))

    return decoded_map, decoded_labels


def preprocess_input(im_file, scale=255.):
    im = cv2.imread(im_file)
    im = np.expand_dims(cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[:, :, 0], -1)
    return im.astype(np.float32) / scale


if __name__ == '__main__':

    # Max value for coefficients
    max_coeffs, _ = max_min_coefficient(quality_range=(50, 100),
                                        n_coeffs=15,
                                        zig_zag_order=True)

    # Load model
    model_file = 'model_QF1_60-98_QF2_90-2-term-loss.h5'
    model = tf.keras.models.load_model(model_file,
                                       custom_objects=({'custom_softmax': custom_softmax_activation(max_coeffs),
                                                        'custom_two_terms_loss_wrapper': custom_two_terms_loss_wrapper(
                                                            max_coeffs, 0.8),
                                                        'custom_mse': custom_mse_wrapper(max_coeffs)})
                                       )

    # Read input image
    input_image = 'T_04_758595.jpg'
    img = preprocess_input(input_image)

    # Create map of size 57x57x15
    map, _ = window_decision(img, model, max_coeffs, (64, 64, 1), stride=1)

    # Save the localization map for the first coefficient
    show_coeffs = [1, 6, 14]
    plt.style.use('classic')
    for n_c in show_coeffs:
        plt.imshow(map[:, :, n_c])
        plt.colorbar()
        plt.savefig(f'{os.path.splitext(os.path.basename(input_image))[0]}_localization_map_coeff{n_c}.png')
        plt.close()

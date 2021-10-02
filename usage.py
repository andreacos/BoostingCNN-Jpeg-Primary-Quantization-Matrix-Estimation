import cv2
import numpy as np
import tensorflow as tf
from networks import custom_two_terms_loss_wrapper, custom_softmax_activation, custom_mse_wrapper
from utils import max_min_coefficient, label2coefficient


def preprocess_input(im_file, target_size, scale=255.):
    im = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)
    if im.shape != target_size:
        im = cv2.resize(im, target_size)

    return im.astype(np.float32) / scale


if __name__ == '__main__':

    model_file = 'models/model_QF1_60-98_QF2_90-2-term-loss.h5'
    # model_file = 'models/model_QF1_55-98_QF2_80-2-term-loss.h5'

    img_file = 'resources/00000000_redaf7d93t.TIF_85_90.png'
    # img_file = 'resources/00000000_redaf7d93t.TIF_50_80.png'

    # Load the table linking each pair of JPEG quality factors to the corresponding Q's coefficients
    qf_map = np.load('resources/qf1_qf2_map_90.npy', allow_pickle=True)
    # qf_map = np.load('resources/qf1_qf2_map_80.npy', allow_pickle=True)

    # Max value for coefficients
    max_coeffs, _ = max_min_coefficient(quality_range=(50, 100),
                                        n_coeffs=15,
                                        zig_zag_order=True)

    model = tf.keras.models.load_model(model_file,
                                       custom_objects=({'custom_softmax': custom_softmax_activation(max_coeffs),
                                                        'custom_two_terms_loss_wrapper': custom_two_terms_loss_wrapper(
                                                            max_coeffs, 0.8),
                                                        'custom_mse': custom_mse_wrapper(max_coeffs)})
                                       )

    x = preprocess_input(img_file, (64, 64), 255.)
    prediction = model.predict(np.expand_dims(x, [0, -1]))

    print(prediction)
    predicted_label = label2coefficient(prediction.flatten(), max_coefficients=max_coeffs)
    print(predicted_label)
    print(len(predicted_label))

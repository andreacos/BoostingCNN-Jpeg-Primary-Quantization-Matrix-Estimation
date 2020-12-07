"""
    2020 Department of Information Engineering and Mathematics, University of Siena, Italy.

    Authors:  Andrea Costanzo (andreacos82@gmail.com) and Benedetta Tondi

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by theFree Software Foundation, either version 3 of the License, or (at your option) any later
    version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details. You should have rec   eived a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.

    If you are using this software, please cite:

    Boosting CNN-based primary quantization matrix estimation of double JPEG images via a classification-like architecture}, 
    Benedetta Tondi and Andrea Costanzo and Dequ Huang and Bin Li
    ArXiv preprint: https://arxiv.org/abs/2012.00468


"""

# ----------------------------------------------------------------------------------------------
# Parameters for dataset creation
# ----------------------------------------------------------------------------------------------

rgb = True                     # If TRUE, coefficients come from Y channel in YCbCr, if FALSE from grayscale
block_size = (64, 64)          # Image size for the whole training / testing process
max_blocks_img = 100           # Number of blocks that are created for each image

file_ext = '.TIF'              # File format for input images (RAISE8K has only TIFF images)
make_train = False             # If FALSE, train dataset is not created (useful for testing new QF pairs)
make_test = True               # If FALSE, test dataset is not created


# ----------------------------------- RAISE ----------------------------------------

# This is the starting folder, from which Train and Test datasets are created
input_train_dir = '/media/hdddati1/Datasets/RAISE8Ksplit/Train'
input_test_dir = '/media/hdddati1/Datasets/RAISE8Ksplit/Test'

# This is the output folder where Train and Test datasets are created
out_train_dir = '/media/hdddati1/Datasets/DeepQuantiFinder/ManyQ/Aligned/Train'
out_test_dir = '/media/hdddati1/Datasets/DeepQuantiFinder/ManyQ/Aligned/Test'

# These are the CSV used to train and test the model. They live in out_train_dir and out_test_dir
training_csv = 'train_many_qf1_qf2_90.csv'
test_csv = 'test_many_qf1_qf2_90.csv'

# ----------------------------------- DRESDEN ----------------------------------------

# This is the starting folder, from which Train and Test datasets are created
# input_train_dir = '/media/hdddati1/Datasets/DeepQuantiFinder/DRESDEN/Train'
# input_test_dir = '/media/hdddati1/Datasets/DeepQuantiFinder/DRESDEN/Test'

# This is the output folder where Train and Test datasets are created
# out_train_dir = '/media/hdddati1/Datasets/DeepQuantiFinder/ManyQ/DRESDEN/Train'
# out_test_dir = '/media/hdddati1/Datasets/DeepQuantiFinder/ManyQ/DRESDEN/Test'

# These are the CSV used to train and test the model. They live in out_train_dir and out_test_dir
# training_csv = 'train_DRESDEN_many_qf1_qf2_90.csv'
# test_csv = 'test_DRESDEN_many_qf1_qf2_90.csv'


# ----------------------------------- PHOTOSHOP ----------------------------------------

# This is the starting folder, from which Train and Test datasets are created
# input_train_dir = '/media/hdddati1/Datasets/DeepQuantiFinder/PHOTOSHOP/Train'
# input_test_dir = '/media/hdddati1/Datasets/DeepQuantiFinder/PHOTOSHOP/Test'

# This is the output folder where Train and Test datasets are created
# out_train_dir = '/media/hdddati1/Datasets/DeepQuantiFinder/ManyQ/PHOTOSHOP/Train'
# out_test_dir = '/media/hdddati1/Datasets/DeepQuantiFinder/ManyQ/PHOTOSHOP/Test'

# These are the CSV used to train and test the model. They live in out_train_dir and out_test_dir
# training_csv = 'train_PHOTOSHOP_qf1_qf2_90.csv'
# test_csv = 'test_PHOTOSHOP_qf1_qf2_90.csv'


# ----------------------------------------------------------------------------------------------
# JPEG parameters
# ----------------------------------------------------------------------------------------------
software = 'python-opencv'              # Software used for JPEG compression
force_jpeg_aligned = False              # If TRUE, only aligned JPEG patches are created
force_jpeg_misaligned = False
zig_zag_order = True                    # If TRUE, JPEG coefficients are always in zig-zag order

# First case-study QF2 = 90
# q_factors = [(60, 90), (65, 90), (70, 90), (75, 90), (80, 90), (85, 90), (90, 90), (95, 90), (98, 90)]

# Second case-study QF2 = 80
# q_factors = [(55, 80), (60, 80), (65, 80), (70, 80), (75, 80), (80, 80), (85, 80), (90, 80), (95, 80)]

# For ALL QF_1 and QF_2 = 90 setup
q_factors = [(60, 90), (61, 90), (62, 90), (63, 90), (64,  90), (65, 90), (66, 90), (67, 90), (68, 90), (69, 90),
             (70, 90), (71, 90), (72, 90), (73, 90), (74,  90), (75, 90), (76, 90), (77, 90), (78, 90), (79, 90),
             (80, 90), (81, 90), (82, 90), (83, 90), (84,  90), (85, 90), (86, 90), (87, 90), (88, 90), (89, 90),
             (90, 90), (91, 90), (92, 90), (93, 90), (94,  90), (95, 90), (96, 90), (97, 90), (98, 90)]


#q_factors = [(60, 92), (61, 92), (62, 92), (63, 92), (64,  92), (65, 92), (66, 92), (67, 92), (68, 92), (69, 92),
#             (70, 92), (71, 92), (72, 92), (73, 92), (74,  92), (75, 92), (76, 92), (77, 92), (78, 92), (79, 92),
#             (80, 92), (81, 92), (82, 92), (83, 92), (84,  92), (85, 92), (86, 92), (87, 92), (88, 92), (89, 92),
#             (90, 92), (91, 92), (92, 92), (93, 92), (94,  92), (95, 92), (96, 92), (97, 92), (98, 92)]

# For ALL QF_1 and QF_2 = 80 setup
#q_factors = [(55, 80), (56, 80), (57, 80), (58, 80), (59, 80),
#             (60, 80), (61, 80), (62, 80), (63, 80), (64, 80), (65, 80), (66, 80), (67, 80), (68, 80), (69, 80),
#             (70, 80), (71, 80), (72, 80), (73, 80), (74, 80), (75, 80), (76, 80), (77, 80), (78, 80), (79, 80),
#             (80, 80), (81, 80), (82, 80), (83, 80), (84, 80), (85, 80), (86, 80), (87, 80), (88, 80), (89, 80),
#             (90, 80), (91, 80), (92, 80), (93, 80), (94, 80), (95, 80), (96, 80), (97, 80), (98, 80)]

# PhotoShop case study QF2 = 90
# q_factors = [(7, 90), (8, 90), (9, 90), (10, 90), (11, 90), (12, 90)]

# PhotoShop case study QF2 = 80
# q_factors_ = [(7, 80), (8, 80), (9, 80), (10, 80), (11, 80), (12, 80)]

n_blocks_train = [1e5]*len(q_factors)
n_blocks_test = [1e4]*len(q_factors)

# ----------------------------------------------------------------------------------------------
# Training / Test parameters
# ----------------------------------------------------------------------------------------------

base_lr = 1e-5                         # Learning rate
max_no_Q_coefs = 15                    # Number of quantisation coefficients used for training
batch_size = 32                        # Training batch size
n_epochs = 50                          # Training epochs
scaling_factor_data = 255.0            # Input images (values [0, 255]) are scaled to [0, 1]
snapshot_frequency = 1000              # Frequency (iterations for saving training metrics)

weighted_label = False
eps_weight = 0.05
mse_weight = 0.8

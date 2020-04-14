
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from nilearn import image
from nilearn import plotting
from nilearn.input_data import NiftiMasker


##--------------------------------------------------------
# DATA LOADING
##--------------------------------------------------------

DATA_PATH = '../data/raw_data/fMRI/hcptrt/sub-01/ses-001/%s'


bold_file = 'sub-01_ses-001_task-wm_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
mask_file = 'sub-01_ses-001_task-wm_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'


masker = NiftiMasker(mask_img= DATA_PATH % mask_file, standardize=True)
series = masker.fit_transform(DATA_PATH % bold_file)

(num_frames, num_channels) = series.shape


# max_value = np.max(series, axis=0)

# normalized_series = (series - max_value)/max_value


##--------------------------------------------------------
# LINEAR MODEL TRAINING
##--------------------------------------------------------

window_size = 12

X = np.array([series[i:i+window_size] for i in range(num_frames - window_size)])
Y = series[window_size:num_frames]

lin_reg = lambda x,y : LinearRegression().fit(x, y)

regressors = [lin_reg(X[:,:,i], Y[:,i]) for i in range(num_channels)]

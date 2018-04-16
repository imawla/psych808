#!/usr/bin/python

# Define function to obtain date and time
def print_now(pre_text=''):
        from datetime import datetime as dt
        now = dt.now().strftime('%Y-%m-%d %H:%M:%S')
        print("{text}{cur_datetime}".format(text=pre_text, cur_datetime=now))
# Print message at script execution
print_now('\nBegin processing searchlight MVPA at ')

##### Initialize variables here
# dataset is the name of the dataset
# cond is a dictionary consisting of contrast names corresponding to copes
#	from second lvl analysis. cond is in the format {condition:copename}
# labelgrp1 and labelgrp2 are two groups for classification 
dataset = 'ds008'
cond = {'go':'cope1', 'stop':'cope2', 'fail stop':'cope3', 'stop-go':'cope4'}
labelgrp1='go'
labelgrp2='stop'
# Print message after script initialization
print("\nSearchlight will compare '{}' and '{}' in '{}' dataset" \
	.format(labelgrp1, labelgrp2, dataset))

##### Import necessary library functions
# Used to read Bash environment variables
from os import getenv
# Used to get the correct expansion of ~, last element of path
from os.path import expanduser, basename
# Used to join directory names into a path with appropriate separator
# characters, even on Windows.
from os.path import join as pathjoin
# Import glob for easy wildcard use
from glob import glob
# Import nibabel for neuroimaging data manipulation
import nibabel as nib
# Import nilearn for MVPA analysis
import nilearn
from nilearn.input_data import NiftiMasker
# Import numpy for array and matrix operations
import numpy as np

# Set the project directory
project_dir = pathjoin(expanduser('~'), 'Projects')
# Set the data directory
data_dir = pathjoin(project_dir, dataset)
# Mask file to conduct searchlight analysis within
#	whole-brain (wb) mask in this case
mask_file = nib.load('/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')

subject_list = glob(pathjoin(data_dir,'sub???','model','task001.gfeat',cond[labelgrp1]+'.feat','stats','cope1.nii.gz')) + \
	glob(pathjoin(data_dir,'sub???','model','task001.gfeat',cond[labelgrp2]+'.feat','stats','cope1.nii.gz'))
subject_list.sort()
subjects = (nib.Nifti1Image.from_image(nib.load(subject)) for subject in subject_list)
subjects_concat = nib.concat_images(subjects, check_affines=False)

fmri_label=[]
subj_label=[]
cond_inv = {v: k for k, v in cond.iteritems()}

for subject in subject_list:
	copenum = subject.split('/')[8].split('.')[0]
	fmri_label = np.append(fmri_label, cond_inv[copenum])
	subjectnum = subject.split('/')[5]
	subj_label = np.append(subj_label, subjectnum)

# Prepare the fMRI data: smooth and apply the mask
from nilearn.input_data import NiftiMasker
nifti_masker = NiftiMasker(mask_img=mask_file,
                           standardize=True, memory='nilearn_cache',
                           memory_level=1)
fmri_masked = nifti_masker.fit_transform(subjects_concat)

# We will perform Support Vector Classification (SVC)
from sklearn.svm import SVC
svc = SVC(kernel = 'linear')

# Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. When doing full-brain analysis, it is better to use
# SelectPercentile, keeping 5% of voxels
# (because it is independent of the resolution of the data).
from sklearn.feature_selection import SelectPercentile, f_classif
feature_selection = SelectPercentile(f_classif, percentile=5)

# We have our classifier (SVC), our feature selection (SelectPercentile),and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])

# Fit decoder and predict
anova_svc.fit(fmri_masked, fmri_label)
prediction = anova_svc.predict(fmri_masked)

#### Obtain prediction scores via cross validation
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score

# Define the cross-validation scheme used for validation.
# Here we use a LeaveOneLabelOut cross-validation on the session label
# which corresponds to a leave-one-session-out
cv = LeaveOneLabelOut(subj_label)

# Compute the prediction accuracy for the different folds (i.e. session)
cv_scores = cross_val_score(anova_svc, fmri_masked, fmri_label, cv=cv)

# Return the corresponding mean prediction accuracy
classification_accuracy = cv_scores.mean()

# Print the results
print("Classification accuracy: %.4f / Chance level: %f" %
      (classification_accuracy, 1. / len(fmri_label.unique())))
# Classification accuracy:  0.70370 / Chance level: 0.5000







# Define the cross-validation scheme used for validation.
# Here we use a KFold cross-validation on the session, which corresponds to
# splitting the samples in 4 folds and make 4 runs using each fold as a test
# set once and the others as learning sets
from sklearn.cross_validation import KFold
cv = KFold(fmri_label.size, n_folds=4)

import nilearn.decoding
# The radius is the one of the Searchlight sphere that will scan the volume
searchlight = nilearn.decoding.SearchLight(
    mask_file,
    radius=5.6, n_jobs=1,
    verbose=1, cv=cv)
searchlight.fit(subjects_concat, fmri_label)

#saving the image files for visualization
from nilearn.image import new_img_like
searchlight_img=new_img_like(mask_file, searchlight.scores_)
nib.save(searchlight_img, pathjoin(project_dir, dataset, 'MVPA_searchlight.nii.gz'))

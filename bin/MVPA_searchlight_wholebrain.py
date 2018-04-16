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
cond_inv = {v: k for k, v in cond.iteritems()}

for subject in subject_list:
	copenum = subject.split('/')[8].split('.')[0]
	#cope_load=nib.load(subject)
	fmri_label=np.append(fmri_label, cond_inv[copenum])
	#cope_masked = wb_mask.fit_transform(subject)
	#fmri_features = np.r_['-1', fmri_features, cope_masked]

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

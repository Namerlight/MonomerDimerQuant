### ASMM_AuSEM dataset

The dataset is organized in the following way. Once you have downloaded the dataset, ensure that the folder structure is correct.

Please contact the authors for access to the dataset.

```
data
│
├─── sample_images\
│    ├─── 001-1-2s.jpeg
│    └─── 3.png
│
├─── correlated\
│    │
│    ├─── bboxes_two_classes\
│    │    └─── json files with bounding boxes for monomers and oligomers for all images
│    │
│    ├─── imageJ_labelled\
│    │    └─── image files with particles detected and labelled using ImageJ's particle detector
│    │
│    ├─── sem_ground_truth\
│    │    └─── original SEM images
│    │
│    ├─── optical_images\
│    │    └─── original optical microscopy images
│    │
│    └─── training_data\
│         ├─── augmented_bin\
│         │    └─── a set of augmented images for reproducibility with binary labels (monomer and oligomers)
│         ├─── augmented_mono\
│         │    └─── a set of augmented images for reproducibility with only particle labels
│         └─── for_training\
│              └─── original optical microscopy images split into train and validation sets with YOLO v8 labels
│     
└─── uncorrelated\
     │
     ├─── optical_uncorrelated\
     │    └─── original white-light excited uncorrelated optical images
     │
     ├─── sliced\
     │    ├─── 001-1-2s\
     │    │    └─── uncorrelated optical images, sliced into 512x512 each.
     │    ├─── ...
     │    └─── 020-1-2s
     │         └───  uncorrelated optical images, sliced into 512x512 each.
     │
     ├─── sliced_processed\
     │    ├─── 001-1-2s\
     │    │    └─── slices images, processed for labelling.
     │    └─── ...
     │
     └─── test\
          ├─── bboxes_test\
          │    └─── json files with bounding boxes for monomers and oligomers for images from sliced_processed
          ├─── test\
          │    ├─── images\
          │    │    └─── images for the test set
          │    └─── labels\
          │         └─── bounding box and object labels for the test set in YOLO v8 format.
          └─── data.yaml
```
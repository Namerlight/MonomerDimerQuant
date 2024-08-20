# Quantifying Monomer-Dimer Distribution of Nanoparticles from Uncorrelated Optical Images

Using Deep Learning to analyze monomers, dimers and higher-order oligomers of gold nanoparticles from optical images.

### Requirements and Setup
Python 3.9 and Pytorch 2.1.2+cu118 have been used.

See `requirements.txt` for other python libraries used.

To install, set up a virtual environment using pip. Then install required libraries using
```shell
pip install -r requirements.txt
```

### Data
Please contact the authors for access to the data used for training the models.

### Models
Download the models from the following link: [Models](https://drive.google.com/file/d/1qhxta9wORrRLUmbf9ifWViVr1obX2eex/view?usp=sharing).

Unzip the two model files and place them into a folder named `models` at the top level of this repository.

Please let me know if the above link does not work.

### Detect and analyze particles.

To count the number of particles in an image of nanoparticles. please call `count_nanoparticles.py`. 
If you have a very large image, the results may be improved by slicing the image. The recommended slice size is 512.
Images will not be sliced if both height and width are less than twice the slice size.

For example, using a sample image path.
```shell
python count_nanoparticles.py -i data\\sample_images\\001-1-2s.jpeg -s 512
```
Any intermediate files created during this process will be stored under `output/imagefilename_timestamp/`

### Visualize Heightmap and Processed Image

To visualize the processed image and the heightmap, please call `generate_visualize_heightmap.py`.

For example, using a sample image path.
```shell
python utils/generate_visualize_heightmap.py -i data/sample_images/3.png
```

## Citation

If you use any of the code, trained models or data, please cite the paper.

TBD
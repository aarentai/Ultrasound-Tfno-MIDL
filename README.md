# Neural Operator Learning for Ultrasound Tomography Inversion
### [Paper](https://arxiv.org/pdf/2304.03297.pdf)
PyTorch implementation of estimating a Riemannian manifold accommodating brain connectome faithfully.<br><br>
 :desert: [Haocheng Dai*](https://users.cs.utah.edu/~haocheng/),
 :desert: [Michael Penwarden*](https://sites.google.com/view/michaelpenwarden),
 :desert: [Mike Kirby](https://users.cs.utah.edu/~kirby/),
 :desert: [Sarang Joshi](https://scholar.google.com/citations?user=GyqdQTEAAAAJ&hl=en) <br>
 :desert:University of Utah
 <br>
Medical Imaging with Deep Learning (MIDL), 2023, Short Paper Track :tent:

<img src='midl2023.png' alt="drawing" width="800"/>

## TL;DR quickstart

To setup a conda environment, begin the training process, and inference:
```
conda env create -f environment.yml
conda activate ultra-tfno
cd Ultrasound-Tfno-MIDL/Scripts/
bash runMetricCnnTrainingInference.sh
```

## Setup

Python 3 dependencies:
```
itk==5.2.0
lazy_import==0.2.2
matplotlib==3.3.1
numba==0.55.1
numpy==1.19.5
PyYAML==6.0
scikit_image==0.18.3
SimpleITK==2.2.1
skimage==0.0
torch==1.10.2
tqdm==4.55.0
```

We provide a conda environment setup file including all of the above dependencies. Create the conda environment `metcnn` by running:
```
conda env create -f environment.yml
```

## What is a Ultrasound TFNO?

An ultrasound TFNO is a tensorized Fourier neural operator trained to estimating a speed of sound field that depict the acoustic preoperties of tissues. The network directly maps from the time-of-flight (TOF) data to a heterogeneous sound speed field, with a single forward pass through the model.This novel application of operator learning circumnavigates the need to solve the computationally intensive iterative inverse problem. It is the first time operator learning has been used for ultrasound tomography and is the first step in potential real-time predictions of soft tissue distribution for tumor identification in beast imaging.

## Citation

```
@misc{dai2023neural,
      title={Neural Operator Learning for Ultrasound Tomography Inversion}, 
      author={Haocheng Dai and Michael Penwarden and Robert M. Kirby and Sarang Joshi},
      year={2023},
      eprint={2304.03297},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
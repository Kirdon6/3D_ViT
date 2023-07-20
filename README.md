# Protein-Ligand Binding Site Detection using 3D Vision Transformers

## Description

The detection of protein-ligand binding sites is not only an important tool for understanding the function of proteins but also finds application in computer-aided drug design in the detection of molecules capable of perturbing protein function. The prediction methods can be divided into sequence-based and structure-based. Similarly, as with images, 3D space can be voxelized, and generalized machine learning methods such as 3D convolutional networks can be used applied for the detection of the binding sites. Recently, a method was published that allows the so-called attention mechanism to be applied to the field of computer vision - Vision Transformer (ViT). The goal of the work is to evaluate the possibilities of extending ViT to 3D, i.e., 3D ViT, for protein-ligand binding sites detection. The work will include a comparison of the implemented approach with existing approaches for protein-ligand binding sites detection.

## Implementation

This is implementation of proposed 3D Vision Transformer in Python. This repositary contains all data and scripts for creating dataset and running Vision Transformer.

## Requirements

- Python 3.11.1

All neccessary packages can be installed via pip

```
pip install -r requirements.txt
```
## Structure
We already provided small version for HOLO4K dataset with complexes with less than 5000 atoms. In order to create own dataset you can run **point_cloud.py** with own parameters. Script **dataset.py** provides class and prepare dataset to correct format for Vision Transformer. It also splits dataset into 3 parts - training, testing and development set. Script **dataset5000.py** creates smaller dataset with complexes with less than 5000 atoms. Script **amino_acid_table.py** consits of properties for amino acids that are used in our model as features. Atomic features are extracted from folder **tables** by **at_table.py**.
The 3D vision transformer is implemented in **vit.py**.

## Usage
Our vision transformer can be runned in command line from the main directory.
```
vit.py
```
This command will run training on our dataset_HOLO4K_small.npz with our default hyperparameters.
You can generate point cloud dataset with **point_cloud.py** by using 
```
point_cloud.py --create-dataset=True --new_file_name==MY_DATASET --input_file=MY_DATASET.npz --proteins_path=DATA --targets_path=TARGETS --protein_list=DATA.ds
```
You can also create dataset from **vit.py** and then run it immediately with using of the same arguments as in previous command.

In order to change hyperparameters, you can do it in command line with flags, e.g.
```
vit.py --batch_size=1
```



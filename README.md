This code package implements the Prototypical Multiple Instance Learning Network (ProtoMIL) by Dawid Rymarczyk,
Aneta Kaczyńska, Jarosław Kraus, Adam Pardyl and Bartosz Zieliński.

This package includes code developped by authors of the paper "This Looks Like That: Deep Learning for Interpretable
Image Recognition" - Chaofan Chen* (Duke University), Oscar Li* (Duke University),Chaofan Tao (Duke University),
Alina Jade Barnett (Duke University), Jonathan Su (MIT Lincoln Laboratory), and Cynthia Rudin (Duke University)
(* denotes equal contribution) as well as of the paper "Attention-based Deep Multiple Instance Learning" - Maximilian Ilse
(University of Amsterdam), Jakub M. Tomczak (University of Amsterdam) and Max Welling (University of Amsterdam).

Licensed under MIT License (see LICENSE for more information regarding the use
and the distribution of this code package).

Python requirements in requirements.txt

# How to run training
* Install requirements from requirements.txt,
* Place dataset files in the data/ folder in the main directory,
* Run `python main.py -d <name of the dataset>`

All possible options can be listed by running `python main.py --help`

# How to run pruning
* Run `python run_pruning.py -d <name of the datset> -l <path to saved training state file>`

# Note on Camelyon16 and TCGA datasets:
Those datasets require additional preprocessing. Code for preprocessing Camelyon16 is located in the datasets/
directory (camelyon_*.py files). A pretrained resnet18 network for preprocessing can be downloaded from
https://github.com/ozanciga/self-supervised-histopathology/releases/tag/tenpercent
For TCGA datsets the data must be preprocessed as for https://github.com/mahmoodlab/CLAM



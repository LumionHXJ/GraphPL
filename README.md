# GraphPL: Leveraging GNN for Efficient and Robust Modalities Imputation in Patchwork Learning
the implementation of GraphPL (based on CLAP).

## introduction for files
* mmnist: dataset, networks and running files for the PolyMNIST dataset.
* mnistsvhntext: dataset, networks and running files for the MNIST-SVHN-TEXT dataset.
* celeba: dataset, networks and running files for the CelebA dataset.
* eicu: dataset, networks and running files for the eICU dataset.
* modalities: modality interface of the datasets.
* eval_metrics: evaluation of the model, including reconstruction MSE, generation coherence, log-likelihood and latent space classification.
* run_script/run_epochs_vaes.py: training script of all methods for all the datasets.
* utils: needed function for the implementation.

## requirements
the needed libraries are in requirements.txt.

## dataset preparation:
The preparation of the PolyMNIST, MNIST-SVHN-TEXT, CelebA datasets follows MoPoE.

### eICU
eICU data set is not directly available and detailed description is in the paper 'Benchmarking machine learning models on eICU critical care dataset'.

## experiments
The scripts for all datasets are provided as `runs/run_{dataset_name}.sh`, with `runs/run_{dataset_name}_graph.sh` specifically for our GraphPL.
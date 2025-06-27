MMADPE: Drug Repositioning Based on Multi-hop Graph Mamba Aggregation with Dual-modality Graph Positional Encoding
Requirements:

    python 3.9.19
    pandas 2.0.3
    cudatoolkit 11.3.1
    pytorch 2.3.0
    numpy 1.23.5
    scikit-learn 1.2.2

Data:

The data files needed to run the model, which contain C-dataset and F-dataset.

    DrugFingerprint, DrugGIP: The similarity measurements of drugs to construct the similarity network
    DiseasePS, DiseaseGIP: The similarity measurements of diseases to construct the similarity network
    DrugDiseaseAssociationNumber: The known drug disease associations
    DiseaseFeature: The features of diseases
    Drug_mol2vec: The features of drugs

Code:

    utils.py: Methods of data processing
    metric.py: Metrics calculation
    train_DDA.py: Train the model

Installation method for the torch_geometric series (install in order):
# pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.3.0+cu121.html
# pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.3.0+cu121.html
# pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-2.3.0+cu121.html
# pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.3.0+cu121.html
# pip install torch-geometric

Usage:

Execute python main.py

# SolPreModel
predicts aqueous solubility via three different embeddings: ECFP, GLEM, FUSED(GLEM+GRAPH)
  - Input: SMILES → ECFP, GLEM, GRAPH
    * Extra Features(for GRAPH): COSMO electron density per grid, WBO per edge, partial charges per atom, molecular quadrupole and Gsolv per graph via xtb calculation
  - Output: LogS

1. Main Dependency
   - python 3.9+
   - pytorch, pytorch-geometric
   - rdkit
   - openbabel
   - numpy / pandas / matplotlib
   - xtb (required for quantum calculation)
  
2. Required Dataset
   - A CSV file that contains a molecule's ID, SMILES, and Experimental LogS. Here AqSolDBc and OChemUnseen was used for train/val and test set respectively.
    
3. What each script does
  - SMILEStoXYZ.py: Converts SMILES string into xyz file. A CSV file that contains a molecule's ID, SMILES, and Exprerimental LogS is needed. Returns .xyz files per molecule. (2D structure → 3D structure)
  - BuildECFPfromSMILES.py: Converts SMILES string into xyz file via rdkit.Chem.MorganGenerator. Returns an npz file containing all the ecfps of the molecules.
  - CreatingGraphEmbedding:
     - PrecomputingCOSMO / PrecomputingWBO: Computes COSMO electron density and WBO via xtb calculation. XYZ file per molecule is needed. Also saves LOG file during the calculation.
     - DataParser: Parses data from cosmo, wbo, log files in the form required for BuildGraphEmbeddings.py.
     - egnn_norm.py: Creates embeddings from graph structure.
     - BuildGraphEmbeddings.py: Builds graph and creates embeddings.
  - PreparingGLEMembedding: read README.md
  - FusionVariants.py: A tool for fusing GLEM and Graph embedding. Provides three different options: simple concatenation, equalmix, and gated fusion.
  - Projection.py: Projector for three different embeddings. Provides PCA, UMAP, Autoencoder, Linear for ECFP and Linear for GLEM and FUSED embedding.
  - train_v2: Contains a regressor, training and plotting module. For ECFP and GLEM training.
  - train_e2e: A end to end script from fusion to training.
  - DataSplit:
    - DataSplitter.py: Splits dataset into train / val / test based on Tanimoto Similarity.
  - Outliers: Extracts common outliers and analyzes predominant features among them. 

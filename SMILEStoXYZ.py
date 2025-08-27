import numpy as np
import csv
from openbabel import pybel
import os

OChem = '/Users/cxxoseo/Desktop/SolPreModel/rawData/OChem_Unseen.csv'
smiles_dict = {}
with open(OChem, newline ='') as f:
    reader = csv.DictReader(f, delimiter=',', quotechar='"')
    for row in reader:
        idx = row['index']
        smiles = row['SMILES']
        smiles_dict[idx] = smiles

for id, smiles in smiles_dict.items():
    mol = pybel.readstring("smi", smiles)
    mol.addh()
    mol.make3D()
    mol.localopt()
    output_path = f"/Users/cxxoseo/Desktop/SolPreModel/OChemUnseentoXYZ/{id}.xyz"
    mol.write("xyz", output_path, overwrite = True)
    print(f"Saved XYZ file to {output_path}")
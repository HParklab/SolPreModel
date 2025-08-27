import numpy as np
import csv
import os

OChem_Unseen = '/Users/cxxoseo/Desktop/SolPreModel/rawData/OChem_Unseen.csv'
INPUT_PATH  = './Desktop/SolPreModel/rawData/'
INPUT = os.path.join(INPUT_PATH, "OChem_Unseen.smi")

os.makedirs(INPUT_PATH, exist_ok=True)

with open(OChem_Unseen, newline ='') as f, open(INPUT, 'w') as f_out:
    reader = csv.DictReader(f, delimiter=',', quotechar='"')
    for row in reader:
        idx = row['index']
        smiles = row['SMILES']
        Sol = row['LogS (Format)']
        line = f"{smiles} {idx} {Sol}\n"
        f_out.write(line)
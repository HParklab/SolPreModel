import sys
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import time
import torch
from gmol.ligand.embedding import pretrained
import multiprocessing as mp
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

model = pretrained.ligand_only_embedder_huggingface('HParklab/ligand-only-embedding')
model.to(device)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

BATCH = 20

input_f = sys.argv[1]
input_type = sys.argv[2] #[smi/sdf]
save_path = sys.argv[3] #.npz

def check_smiles(input_f):
    args = []
    for i,l in enumerate(open(input_f)):
        words = l[:-1].split()
        if len(words) < 2: continue
        smi = words[0]
        tag = words[1]
        args.append((smi,tag))

    def run_single(args):
        smi,tag = args
        mol = Chem.MolFromSmiles(smi)
        try:
            mol = Chem.AddHs(mol)
            status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            mol.GetConformer().GetPositions()
        except:
            return False
        return smi,tag

    smis_valid = []
    tags_valid = []
    #a = mp.Pool(processes=10)
    #ans = a.map(run_single,args)
    #for an in ans:
    #    if an:
    #        print(an)
    #        smis_valid.append(an[0])
    #        tags_valid.append(an[1])
    
    for arg in args:
        out = run_single(arg)
        if out:
            print(arg[0],arg[1])
            smis_valid.append(out[0])
            tags_valid.append(out[1])
    print(f"valid smiles: {len(smis_valid)}/{i+1}")
    return smis_valid, tags_valid

def read_sdf(sdf):
    
    #reader = Chem.SDMolSupplier(sdf)
    #for mol in reader:
        #xyz = mol.GetConformer().GetPositions()
        #for i,atom in enumerate(mol.GetAtoms()):
        #    print(atom.GetAtomicNum(), xyz[i])
    return
    
# Example SMILES dataset
if input_type == 'smi':
    print("checking smiles...")
    inputs,tags = check_smiles(input_f)
    converter = pretrained.get_smiles_batch_converter()
elif input_type == 'sdf':
    inputs = [l[:-1] for l in open(input_f)]
    tags = [a.split('/')[-1].split('.sdf')[0] for a in inputs]
    converter = pretrained.get_sdf_batch_converter()

# Extract ligand-only embeddings
## split by batchsize=20
nbatch = int((len(inputs)-1)/BATCH)+1
atom_embs = {}
smiles = {}
print("running %d inputs in %d batch"%(len(inputs), nbatch))
for i in range(nbatch):
    if i == nbatch-1:
        input_batch = inputs[i*BATCH:]
        tags_batch = tags[i*BATCH:]
    else:
        input_batch = inputs[i*BATCH:(i+1)*BATCH]
        tags_batch = tags[i*BATCH:(i+1)*BATCH]
    if input_batch == []: break

    if isinstance(input_batch,str): input_batch = [input_batch]

    try:
    #if True:
        print(input_batch)
        g2d, g3d, batch_metadata, batch_physichem = converter(input_batch,device=device)
        '''
        g2d.to(device)
        g3d.to(device)
        for a in batch_metadata:
            if hasattr(a,'to'): a.to(device)
        
        for a in batch_metadata:
            if hasattr(a,'to'): print(a.device)
        #batch_metadata.to(device)
        #batch_physichem.to(device)
        '''
        
        ( # (N_batch, N_atoms, embedding_dim) # (N_batch, N_atoms, N_atoms, embedding_dim)
            emb_atom_single, emb_atom_pair, emb_frag_single, emb_frag_pair,
            (atom_2d_n_mask, atom_2d_e_mask),
            (frag_2d_n_mask, frag_2d_e_mask),
        ) = model(g2d, g3d, batch_metadata, batch_physichem)
    except:
    #else:
        print("failed to launch", input_batch)
        continue

    for a,tag,emb,mask in zip(input_batch, tags_batch, emb_atom_single, atom_2d_n_mask):
        emb_trimmed = emb[torch.where(mask>0)[0]].cpu().detach().numpy()
        atom_embs[tag] = emb_trimmed
        if input_type == 'smi':
            smiles[tag] = a
    
    print(f"done {i+1}/{nbatch} batches with batchsize={BATCH}...")

np.savez(save_path, emb=atom_embs, smiles=smiles)


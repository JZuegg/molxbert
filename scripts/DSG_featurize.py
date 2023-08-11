#!/root/miniconda3/envs/molbert/bin/python
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
import csv
import os
from array import array
from contextlib import suppress
import pandas as pd

ChkDir = "D:/Code/MolBERT/data"
path_to_checkpoint = os.path.join(ChkDir,'molbert300','checkpoints','last.ckpt')
#path_to_checkpoint = os.path.join(ChkDir,'molbert_100epochs','checkpoints','last.ckpt')
#path_to_checkpoint = './data/molbert_100epochs/checkpoints/last.ckpt'

#path_to_checkpoint = '../neoMolBERT_10epochs/checkpoints/neoMolBERT264_10epoch.ckpt' #this is my model
f = MolBertFeaturizer(path_to_checkpoint,device='cuda') 
print(f.__getstate__())


CsvDir = "D:/Code/GBM/CDD/Output"
CsvSmiles = os.path.join(CsvDir,'GBM_CDD_MolInfo.csv')
MolBVector = os.path.join(CsvDir,'GBM_CDD_MolBert300_2.csv')
df = pd.read_csv(CsvSmiles)
molB = {}                      

#with open(CsvSmiles) as fcsv: #load smiles from csv into list
for idx,row in df.iterrows():
        if ~pd.isnull(row['SMILES']):
            molid = row['MolID']
            smi = row['SMILES']
            features, masks = f.transform([smi])
            try:
                features, masks = f.transform([smi])
                molB[molid] = features[0]
                print(f"{molid}: {len(features[0])}")
            except:
                print(f"{molid}: None")
                molB[molid] = 0

molB_df = pd.DataFrame(molB).T
molB_df.to_csv(MolBVector) 
#with open('out.csv','a') as f:
#    for d in data:
#        #print(','.join([str(i) for i in d])+'\n')
#        f.write(','.join([str(i) for i in d])+'\n')

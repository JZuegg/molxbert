#
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
import csv
import os

DataDir = "/home/uqjzuegg/scratch/Data"
ChkDir = os.path.join(DataDir,'MolBert','pretrained','molbert_100epochs') 
path_to_checkpoint = os.path.join(ChkDir,'checkpoints','last.ckpt')

CsvDir = os.path.join(DataDir,'COADD')
CsvSmiles = os.path.join(CsvDir,'test.smi')
MolBVector = os.path.join(CsvDir,'test_MB128.csv')

print(f"--------------------------------------------------")
print(f" ..START.. ")
print(f"--------------------------------------------------")
print(f" {ChkDir}")
print(f" {path_to_checkpoint}")
print(f" {CsvDir}")
print(f" {CsvSmiles}")
print(f" {MolBVector}")

f = MolBertFeaturizer(path_to_checkpoint,device='cuda') 
print(f" {f.__getstate__()}")

nProc = {'Read':0, 'InValid':0}
print(f"--------------------------------------------------")

with open(CsvSmiles) as fsmi: 
    smi_reader = csv.reader(fsmi,delimiter=',')
    with open(MolBVector, mode = 'w') as fvec:
        csv_writer = csv.writer(fvec,delimiter=',')
        for row in smi_reader:
            nProc['Read'] += 1
            try:
                features, masks = f.transform(row[0])
                csv_writer.writerow(features[0])
            except:
                print(f"{row[0]}: None")
                nProc['InValid'] += 1
print(f"--------------------------------------------------")
print(f" {ChkDir}")
print(f" {path_to_checkpoint}")
print(f" {CsvDir}")
print(f" {CsvSmiles}")
print(f" {MolBVector}")
print(f" {nProc}")
print(f"--------------------------------------------------")
print(f" ..END.. ")
print(f"--------------------------------------------------")

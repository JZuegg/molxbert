#!/bin/bash
#SBATCH --job-name=MBerTorch_TR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --output log/molbert_train_out.txt
#SBATCH --error log/molbert_train_error.txt

# Initialise Conda Env
module load anaconda
source /opt/ohpc/pub/apps/anaconda3/etc/profile.d/conda.sh
conda activate pytorch2

# Initialise MolBert 
source /clusterdata/uqjzuegg/scr/zpyCode/03_Apps/zChemBERT/molbert/scripts/init_molbert.sh

srun python molbert/apps/smiles.py \
    --train_file /clusterdata/uqjzuegg/scr/Data/Guacamol/guacamol_v1_train.smiles \
    --valid_file /clusterdata/uqjzuegg/scr/Data/Guacamol/guacamol_v1_valid.smiles \
    --max_seq_length 128 \
    --batch_size 16 \
    --masked_lm 1 \
    --num_physchem_properties 200 \
    --is_same_smiles 0 \
    --permute 1 \
    --max_epochs 20 \
    --num_workers 4 \
    --val_check_interval 1

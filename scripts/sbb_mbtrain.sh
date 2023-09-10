#!/bin/bash
#SBATCH --job-name=mxb_train
#SBATCH --partition=gpu_cua
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --account=a_blaskovich
#SBATCH --output log/molbert_train_out.txt
#SBATCH --error log/molbert_train_error.txt

# Initialise Conda Env (Bunya)
export PYTHONPATH=/home/uqjzuegg/scratch/zpyCode/01_Library/zLib
module load anaconda3
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate pytorch2

# Initialise MolBert 
source /home/uqjzuegg/BlaskLab_Micro/Calc/pyMolxBert/molxbert/scripts/init_molbert.sh

srun python molbert/apps/smiles.py \
    --train_file /home/uqjzuegg/BlaskLab_Micro/databases/gucamol/guacamol_v1_train.smiles \
    --valid_file /home/uqjzuegg/BlaskLab_Micro/databases/gucamol/guacamol_v1_valid.smiles \
    --max_seq_length 128 \
    --batch_size 32 \
    --masked_lm 1 \
    --num_physchem_properties 200 \
    --is_same_smiles 0 \
    --permute 1 \
    --max_epochs 20 \
    --num_workers 16 \
    --val_check_interval 1

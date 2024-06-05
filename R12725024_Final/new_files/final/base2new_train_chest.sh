#!/bin/bash

# cd ../..

# custom config
# DATA=/path/to/datasets
DATA=/home/arthur/hw/dlmi/all/CoOp/data
TRAINER=CoCoOp
# TRAINER=CoOp

DATASET=$1
SEED=$2

CFG=vit_b16_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=1000000

# nodulepneumonia cardiomegaly hilar_enlargement fracture_old fibrosis aortic_calcification tortuous_aorta thickened_pleura TB pneumothorax emphysema atelectasis calcification pulmonary_edema increased_lung_markings elevated_diaphragm consolidation

for TARGET in pleural_effusion nodule pneumonia cardiomegaly hilar_enlargement fracture_old fibrosis aortic_calcification tortuous_aorta thickened_pleura TB pneumothorax emphysema atelectasis calcification pulmonary_edema increased_lung_markings elevated_diaphragm consolidation
do
    for SHOT in 1 5 10
    do
        # rm -rf data/medfmc/MedFMC_train/split_tsai_medfmc_train_chest.json
        rm -rf data/medfmc/MedFMC/chest/split_fewshot
        rm -rf data/medfmc/MedFMC/colon/split_fewshot
        rm -rf data/medfmc/MedFMC/endo/split_fewshot
        # rm -rf output/base2new/train_base/medfmc_train_chest
        
        DIR=output/base2new/train_base/${DATASET}/shots_${SHOT}_target_${TARGET}/${TRAINER}/${CFG}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES all \
            DATASET.TARGET ${TARGET} \
            DATASET.NUM_SHOT ${SHOT}
        fi
        
        # mv data/medfmc/MedFMC_train/chest/images/${TARGET}\ negative/* data/medfmc/MedFMC_train/chest/images/
        # mv data/medfmc/MedFMC_train/chest/images/${TARGET}\ positive/* data/medfmc/MedFMC_train/chest/images/
    
        # mv output/base2new/train_base/medfmc_train_chest output/base2new/train_base/medfmc_train_chest-${TARGET}
    
        # rm -rf data/medfmc/MedFMC_train/chest/images/${TARGET}\ negative
        # rm -rf data/medfmc/MedFMC_train/chest/images/${TARGET}\ positive
    done
done

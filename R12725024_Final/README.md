# Challenge
https://medfm2023.grand-challenge.org/medfm2023/

# Initialization

export codePath="/home/arthur/hw/dlmi/all"
mkdir -p $codePath
export dataPath="/service/amy2/arthur/MedFMC"
mkdir -p $dataPath
cd $codePath



# mmpretrain installation - recommended
https://mmpretrain.readthedocs.io/en/latest/get_started.html

conda create --name openmmlab python=3.8 -y
conda activate openmmlab
<!--conda install pytorch torchvision -c pytorch-->
pip3 install torch torchvision torchaudio
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
pip install -U openmim && mim install -e .
pip install -U openmim && mim install "mmpretrain>=1.0.0rc8" -y
pip install colorama
pip install lxml
pip install regex
pip install gdown

# mmpretrain installation - inferior
(https://github.com/open-mmlab/mmpretrain/tree/17a886cb5825cd8c26df4e65f7112d404b99fe12?tab=readme-ov-file)

<!--conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y-->
<!--conda activate open-mmlab-->
<!--pip install openmim-->
<!--git clone https://github.com/open-mmlab/mmpretrain.git-->
<!--cd mmpretrain-->
<!--mim install -e .-->

# Data Preparation

cd $dataPath
gdown https://drive.google.com/uc?id=1a4P2Twh7ZCSwS4aPsOZtYX3Y-ClQHMBM
gdown https://drive.google.com/uc?id=1LsEYcJZj_5tkyvUsgjnI-81e1plmuh9D

cd $codePath/mmpretrain

ln -s $dataPath data
cd data

mkdir medfmc
cd medfmc

<!--mkdir data_backup-->

<!--upload data_backup to $codePath-->
[data_backup](https://github.com/openmedlab/MedFM/tree/main/data_backup)
mv $codePath/data_backup $codePath/mmpretrain/data/medfmc/data_backup
gdown https://drive.google.com/uc?id=1a4P2Twh7ZCSwS4aPsOZtYX3Y-ClQHMBM
gdown https://drive.google.com/uc?id=1LsEYcJZj_5tkyvUsgjnI-81e1plmuh9D
unzip MedFMC_train_v2.zip
unzip MedFMC_val_v2.zip

<!--mkdir -p MedFMC-->
```bash
for target in chest colon endo
do
    mkdir -p MedFMC/${target}/images
    mkdir -p MedFMC/${target}/images
    mkdir -p MedFMC/${target}/images

    for split in train val
    do
        mv MedFMC_${split}/${target}/${target}_${split}.csv MedFMC/${target}/
        mv MedFMC_${split}/${target}/images/* MedFMC/${target}/images/
    done
done
```


cp $codePath/new_files/gen_ann_file_v2.ipynb ./

execute gen_ann_file_v2.ipynb under data/medfmc/

```bash
cp $codePath/new_files/hw/densenet121_4xb256_in1k-chest.py $codePath/mmpretrain/configs/densenet/
cp $codePath/new_files/hw/densenet121_4xb256_in1k-colon.py $codePath/mmpretrain/configs/densenet/
cp $codePath/new_files/hw/densenet121_4xb256_in1k-endo.py $codePath/mmpretrain/configs/densenet/

cp $codePath/new_files/hw/densenet121-multilabel.py $codePath/mmpretrain/configs/_base_/models/densenet/

cp $codePath/new_files/hw/imagenet_bs64-chest.py $codePath/mmpretrain/configs/_base_/datasets/
cp $codePath/new_files/hw/imagenet_bs64-colon.py $codePath/mmpretrain/configs/_base_/datasets/
cp $codePath/new_files/hw/imagenet_bs64-endo.py $codePath/mmpretrain/configs/_base_/datasets/

cp $codePath/new_files/hw/train.sh $codePath/mmpretrain/
cp $codePath/new_files/hw/test.sh $codePath/mmpretrain/

cd $codePath/mmpretrain
```

mkdir pretrain
cd pretrain
wget https://download.openmmlab.com/mmclassification/v0/densenet/densenet121_4xb256_in1k_20220426-07450f99.pth
(== Download [densenet121_4xb256_in1k_20220426-07450f99.pth](https://download.openmmlab.com/mmclassification/v0/densenet/densenet121_4xb256_in1k_20220426-07450f99.pth) from https://mmpretrain.readthedocs.io/en/latest/papers/densenet.html under folder pretrain)


cd ..

bash train.sh
bash test.sh

rm -rf pymp-*

cp $codePath/new_files/evaluate_densenet.ipynb $codePath/mmpretrain
execute evaluate_densenet.ipynb to see the evaluation




















<!--mv $codePath/new_files/generate_custom_label.ipynb generate_custom_label.ipynb-->
<!--mv $codePath/new_files/final/gen_ann_file.ipynb gen_ann_file.ipynb-->



cd $codePath

# CoCoOp: https://arxiv.org/pdf/2203.05557
## Installation: https://github.com/KaiyangZhou/CoOp?tab=readme-ov-file
1. Follow the instruction to install Dassl: https://github.com/KaiyangZhou/Dassl.pytorch#installation
```bash
# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Create a conda environment
conda create -y -n dassl python=3.8

# Activate the environment
conda activate dassl

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
<!--conda install pytorch torchvision cudatoolkit=10.2 -c pytorch-->
pip3 install torch torchvision torchaudio

<!--pip install colorama-->
<!--pip install lxml-->
<!--pip install regex-->

# Install dependencies
pip install -r requirements.txt



# Install this library (no need to re-build if the source code is modified)

rm -rf $codePath/Dassl.pytorch/dassl/evaluation/evaluator.py
cp $codePath/new_files/evaluator.py $codePath/Dassl.pytorch/dassl/evaluation/

python setup.py develop


```
2. Install CoCoOp
```bash
cd $codePath
git clone https://github.com/KaiyangZhou/CoOp.git
cd CoOp
pip install -r requirements.txt
If you want to check whether it can work in your env: Follow [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to install the datasets.

e.g. wget https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1 to install caltech data

```

3. Start
```bash

ln -s $dataPath data ( = ln -s /service/amy2/arthur/MedFMC data )

cp $codePath/new_files/final/base2new_train_chest.sh $codePath/CoOp/scripts/cocoop/
cp $codePath/new_files/final/base2new_train_colon.sh $codePath/CoOp/scripts/cocoop/
cp $codePath/new_files/final/base2new_train_endo.sh $codePath/CoOp/scripts/cocoop/

cp $codePath/new_files/final/medfmc_chest.py $codePath/CoOp/datasets/
cp $codePath/new_files/final/medfmc_colon.py $codePath/CoOp/datasets/
cp $codePath/new_files/final/medfmc_endo.py $codePath/CoOp/datasets/

cp $codePath/new_files/final/medfmc_chest.yaml $codePath/CoOp/configs/datasets/
cp $codePath/new_files/final/medfmc_colon.yaml $codePath/CoOp/configs/datasets/
cp $codePath/new_files/final/medfmc_endo.yaml $codePath/CoOp/configs/datasets/

cp $codePath/new_files/final/train.py $codePath/CoOp/
cp $codePath/new_files/final/zsclip.py $codePath/CoOp/trainers/

bash scripts/cocoop/base2new_train_chest.sh medfmc_chest 1
bash scripts/cocoop/base2new_train_colon.sh medfmc_colon 1
bash scripts/cocoop/base2new_train_endo.sh medfmc_endo 1


```


cp $codePath/new_files/evaluate_CoCoOp.ipynb $codePath/CoOp/
execute evaluate_CoCoOp.ipynb to see the evaluation


# Future Work
Apply the following research:
1. DualCoOp: https://arxiv.org/pdf/2206.09541
2. TAI: https://arxiv.org/pdf/2211.12739
3. TAI++: https://arxiv.org/pdf/2405.06926v1


mv MedFM/medfmc mmpretrain/
export PYTHONPATH=$PWD:$PYTHONPATH


python tools/train.py configs/swin_b-vpt/in21k-swin-b_vpt5_bs4_lr5e-2_1-shot_chest_adamw.py

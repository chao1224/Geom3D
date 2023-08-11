# Specifications of Dataset Download in Geom3D

We provide both the raw and processed data at [this HuggingFace link](https://huggingface.co/datasets/chao1224/Geom3D_data).

## PCQM4Mv2

```
mkdir -p pcqm4mv2/raw
cd pcqm4mv2/raw
wget http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz
tar -xf pcqm4m-v2-train.sdf.tar.gz


wget http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip
unzip pcqm4m-v2.zip
mv pcqm4m-v2/raw/data.csv.gz .
rm pcqm4m-v2.zip
rm -rf pcqm4m-v2 
```

## GEOM

```
wget https://dataverse.harvard.edu/api/access/datafile/4327252
mv 4327252 rdkit_folder.tar.gz
tar -xvf rdkit_folder.tar.gz
```

## Molecule3D

Install it following the google drive link [here](https://github.com/divelab/MoleculeX/tree/molx/Molecule3D).

## QM9

Automatically installed under folder `.QM9/raw`.

## MD17

Automatically installed under folder `./MD17`.

In March 2023 (or even earlier), they updated the MD17 FTP site, and the previous datasets are missing. We may need to keep and upload a version to the website.

## rMD17

Download the dataset from [this link](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038?file=24013628), and put the file `12672038.zip` under `./rMD17` folder.
- `unzip 12672038.zip`
- `tar xjf rmd17.tar.bz2`
- `mv rmd17/npz_data .`
- `mv rmd17/splits .`

## COLL

We use this repo: `git@github.com:TUM-DAML/gemnet_pytorch.git`.

## LBA/PDBBind

```
mkdir -p lba/raw
mkdir -p lba/processed
cd lba/raw
# wget http://www.pdbbind.org.cn/download/pdbbind_v2015_refined_set.tar.gz
# wget http://www.pdbbind.org.cn/download/pdbbind_v2018_refined.tar.gz
# wget http://www.pdbbind.org.cn/download/pdbbind_v2019_refined.tar.gz
# wget https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-30-indices.tar.gz

wget http://www.pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz
tar -xzvf PDBbind_v2020_refined.tar.gz

wget https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-30.tar.gz
tar -xzvf LBA-split-by-sequence-identity-30.tar.gz
mv split-by-sequence-identity-30/indices ../processed/
mv split-by-sequence-identity-30/targets ../processed/
```

## LEP

```
mkdir -p lep/raw
mkdir -p lep/processed
cd lep/raw

wget https://zenodo.org/record/4914734/files/LEP-raw.tar.gz
tar -xzvf LEP-raw.tar.gz
wget https://zenodo.org/record/4914734/files/LEP-split-by-protein.tar.gz
tar -xzvf LEP-split-by-protein.tar.gz
```


## MoleculeNet dataset

```
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
dataset_list=(tox21 toxcast clintox bbbp sider muv hiv bace)
for dataset in "${dataset_list[@]}"; do
    mkdir -p molecule_datasets/"$dataset"/raw
    cp dataset/"$dataset"/raw/* molecule_datasets/"$dataset"/raw/
done
rm -rf dataset

wget -O malaria-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/malaria-processed.csv
mkdir -p ./molecule_datasets/malaria/raw
mv malaria-processed.csv ./molecule_datasets/malaria/raw/malaria.csv

wget -O cep-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-02-cep-pce/cep-processed.csv
mkdir -p ./molecule_datasets/cep/raw
mv cep-processed.csv ./molecule_datasets/cep/raw/cep.csv
```

## EC & FOLD
Check this [link](https://github.com/phermosilla/IEConv_proteins#download-the-preprocessed-datasets).

- `ProtFunct` is for task `EC`
- `HomologyTAPE` is for task `FOLD`

Or
- `cd EC; python download.py`
- `cd FOLD; python download.py`

## MatBench

```
mkdir MatBench
cd MatBench

wget https://figshare.com/ndownloader/files/17494820
mv 17494820 expt_is_metal.json.gz
gzip -d expt_is_metal.json.gz

wget https://figshare.com/ndownloader/files/17494814
mv 17494814 expt_gap.json.gz
gzip -d expt_gap.json.gz

wget https://figshare.com/ndownloader/files/17494637
mv 17494637 glass.json.gz
gzip -d glass.json.gz

wget https://figshare.com/ndownloader/articles/9755486/versions/2
mv 2 perovskites.json.gz
unzip perovskites.json.gz
rm perovskites.json.gz
rm 17494805_perovskites.json.gz
gzip -d 17494808_perovskites.json.gz
mv 17494808_perovskites.json perovskites.json

wget https://figshare.com/ndownloader/files/17476067
mv 17476067 dielectric.json.gz
gzip -d dielectric.json.gz

wget https://figshare.com/ndownloader/files/17476064
mv 17476064 log_gvrh.json.gz
gzip -d log_gvrh.json.gz

wget https://figshare.com/ndownloader/files/17476061
mv 17476061 log_kvrh.json.gz
gzip -d log_kvrh.json.gz

wget https://figshare.com/ndownloader/files/17476046
mv 17476046 jdft2d.json.gz
gzip -d jdft2d.json.gz

wget https://figshare.com/ndownloader/files/17476040
mv 17476040 steels.json.gz
gzip -d steels.json.gz

wget https://figshare.com/ndownloader/files/17476037
mv 17476037 phonons.json.gz
gzip -d phonons.json.gz

wget https://figshare.com/ndownloader/files/17476034
mv 17476034 mp_is_metal.json.gz
gzip -d mp_is_metal.json.gz

wget https://figshare.com/ndownloader/files/17476028
mv 17476028 mp_e_form.json.gz
gzip -d mp_e_form.json.gz

wget https://figshare.com/ndownloader/files/17084741
mv 17084741 mp_gap.json.gz
gzip -d mp_gap.json.gz
```

The dataset size can match with [MatBenchmark v0.1](https://github.com/materialsproject/matbench/blob/main/matbench/matbench_v0.1_dataset_metadata.json).

## QMOF

```
mkdir QMOF
cd QMOF
wget https://figshare.com/ndownloader/articles/13147324/versions/13
mv 13 qmof_database_v13.zip
unzip qmof_database_v13.zip
unzip qmof_database.zip

cd qmof_database
python xyz_to_cifs.py
cd ../..
```

Or follow [this link](https://github.com/arosen93/QMOF/blob/main/benchmarks.md) for prediction on QMOF DB v13.


# Weighted Graph Embedding
This repository includes the implementation for Link Prediction of Weighted Triples for Knowledge Graph Completion within the Scholarly Domain, submitted to IEEE Access Journal.

**Datasets**
- [x] AIDA35k
- [x] NL27k
- [x] CN15k
- [x] PPI5k

**Models**
- [x] WGE_logi
- [x] WGE_rect
- [x] UKGE_logi
- [x] UKGE_rect
- [x] distmult
- [x] complEx
- [x] transE

## Installation

Downloading the repository 
```
git clone https://github.com/gokcemuge/WeightedGraphEmbedding.git
```

## Running the code
Example command for running the model WGE_rect 
```
python train.py --dataset aida35k --model WGE_rect -groundings false -rule_coef 0.5 -reg_scale 0.001 -dim 128 -lr 0.0001 -b 512 -neg 10 -val_e 50 -test_e 100 -max 100 -load false -save false
```
Example command for running the model WGE_rect with rules
```
python train.py --dataset aida35k --model WGE_rect -groundings true -rule_coef 0.5 -reg_scale 0.001 -dim 128 -lr 0.0001 -b 512 -neg 10 -val_e 50 -test_e 100 -max 100 -load false -save false
```
Example command for running the model ComplEx
```
python train.py --dataset aida35k --model complEx -margin 4.0 -dim 128 -lr 0.0001 -b 512 -neg 10 -plot false -val_e 40 -test_e 80 -max 80 -load false -save false

```
Example command for running the model UKGE_logi with psl
```
python train.py --dataset nl27k --model UKGE_logi -psl true -dim 128 -b 512 -lr 0.001 -neg 10 -plot false -val_e 50 -test_e 100 -max 100 
```

AIDA35k is created from Academy Industry Dynamics Knowledge Graph [AIDA](https://aida.kmi.open.ac.uk/).

NL27k, CN15k and PPI5k are provided by [UKGE](https://github.com/stasl0217/UKGE).
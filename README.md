# JL_Analysis
Valerie and Joel, dimensionally reducing 

## To load Sparsification_Research Repository Run
`git submodule update --init --recursive`

## To update Sparsification_Research Directory
`cd Sparsification_Research`
`git pull origin main`

## To run main:
python -m src.main

## TODO
- Make JL take in just epsilon, d is determined by the following: d = (C \ln n) / \epsilon^2
- Find apt C, from this: d = (C \ln n) / \epsilon^2
- Check top right eigenvector preservation
- Behavior on Sparse matrices
- Inverse of Johnson Lindenstrauss: Moore-Penrose Psuedo-Inverse -> Singular Value Decomposition

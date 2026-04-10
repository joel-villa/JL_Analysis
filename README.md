# JL_Analysis
We dimensionally reduce

## To load Sparsification_Research Repository Run
`git submodule update --init --recursive`

## To update Sparsification_Research Directory
`cd Sparsification_Research`
`git pull origin main`

## To run main:
python -m src.main

## TODO

### High Priority
- Implement cosine distance (dot product)
- Debug Tester (eigenvector non-normalized - maybe see if there is numpy built in functions for finding top right/left eigenvectors)
- Check top right eigenvector preservation
- Behavior on Sparse matrices

### Mid Priority
- From rectangular to square? 
- Make JL take in just epsilon, d is determined by the following: d = (C \ln n) / \epsilon^2
- Make Tests for preservation of Dot Products, Distance Vectors, etc
- Find apt C, from this: d = (C ln n) / epsilon^2
  
### Low Priority
- Inverse of Johnson Lindenstrauss: Moore-Penrose Psuedo-Inverse -> Singular Value Decomposition
- Test the modified version of Thm 8.2.2 from https://users.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/8-sparsification.pdf, s.t. expected density is maintained
  

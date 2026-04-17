# JL_Analysis
We dimensionally reduce

## To load Sparsification_Research Repository Run
`git submodule update --init --recursive`

## To update Sparsification_Research Directory
`cd Sparsification_Research`
`git pull origin main`

## To run main:
python -m src.main

## A Note on Cosine Distances

Cosine distances are from 0 to 2, where 0 is identical, 1 is orthogonal, and 
2 is opposite

## TODO

### High Priority
- Power iteration w/ JL reductions, Sparsified, and...
    - Use subset matrices itteratively (akin to boosting) - random sampling
- Behavior on various Sparse matrices

### Mid Priority

- Make Tests for preservation of Dot Products, Distance Vectors, etc
- Find apt C, from this: d = (C ln n) / epsilon^2
  
### Low Priority
- Inverse of Johnson Lindenstrauss: Moore-Penrose Psuedo-Inverse -> Singular Value Decomposition
- Test the modified version of Thm 8.2.2 from https://users.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/8-sparsification.pdf, s.t. expected density is maintained (Connor if you want to mess arround with this, just lmk, I have it coded up already, just in a seperate repo)
- Make JL take in just epsilon, d is determined by the following: d = (C \ln n) / \epsilon^2
  

# Environment
```
conda env create -f environments/<env-name>.yml
conda activate encoder-addition
```

# Results

### Plain Addition Decoder Model 90000 training examples
- Converged by 3000 iterations
- Final train and test loss was around 1.1
- 9474/10000 correct on test dataset, 94.74% accuracy

### Plain Addition Decoder Model 20000 training examples
- Converged by 3000 iterations
- Final train and test loss was around 1.1
- 45293/50000 correct on test dataset, 90.59% accuracy

### Plain Addition Decoder Model 15000 training examples
- Converged by 3000 iterations
- Final train and test loss was around 1.1
- 42004/50000 correct on test dataset, 84.01% accuracy

### Plain Addition Decoder Model 10000 training examples
- Converged and overfit by 3000 iterations
- Best test loss was around 1.2
- 24143/50000 correct on test dataset, 48.29% accuracy

### Plain Addition Encoder Model 90000 training examples
- Converged by 3000 iterations
- Final train and test loss was around 1.1
- 9666/10000 correct on test dataset, 96.66% accuracy

### Plain Addition Decoder Model 15000 training examples
- Converged by 3000 iterations
- Final train and test loss was around 1.1
- 45140/50000 correct on test dataset, 90.28% accuracy
# Environment
### Standard (Training and Evaluation)
```
conda env create -f env.yml
conda activate encoder-addition
```
### Devolopement (Training, Evaluation, and Testing)
```
conda env create -f env-dev.yml
conda activate encoder-addition-dev
```

# Results

### Plain Addition Decoder Model 90000 training examples
- Converged by 3000 iterations
- Final train and test loss was around 1.01
- 99676/100000 correct on test dataset

### Reversed Addition Decoder Model 90000 training examples
- Converged by 3000 iterations
- Final train and test loss was around 1.01
- 97827/100000 correct on test dataset
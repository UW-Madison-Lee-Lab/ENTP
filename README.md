# Environment
```
conda env create -f environments/<env-name>.yml
conda activate encoder-addition
```

# Results

| Experiment | Best Validation Loss | Best Validation Loss Iteration | Testing Results | Test Accuracy |
|----------|----------|----------|----------|----------|
| Plain Addition Decoder Model 15000 training examples | 1.08 | 2700 | 42036/50000 | 84.07% |
| Plain Addition Decoder Model 15000 training examples | 1.07 | 2700 | 45861/50000 | 91.72% |
| Reversed Addition Decoder Model 15000 training examples | 1.11 | 2900 | 38290/50000 | 76.58% |
| Reversed Addition Decoder Model 15000 training examples | 1.07 | 2700 | 46943/50000 | 93.89% |

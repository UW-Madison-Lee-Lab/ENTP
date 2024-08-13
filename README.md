# Environment
```bash
conda env create -f environments/<env-name>.yml
conda activate encoder-addition
```

# AWS
- amazon/Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

exec $SHELL

sudo apt install git-all
sudo apt install tmux
```
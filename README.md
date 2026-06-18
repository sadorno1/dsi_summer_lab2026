# dsi_summer_lab2026

# Environment Setup

## HPC (Midway3)

Shared conda environment located at `/project/ycleong/users/dsiSummerLab26/env`

### One time only

The first two commands make `conda activate` work in your shell. The third tells conda to look for environments in the shared folder.

```bash
conda init bash
source ~/.bashrc
conda config --add envs_dirs /project/ycleong/users/dsiSummerLab26/env
```

### Every session

```bash
module load python/anaconda-2023.09
conda activate naturalistic
```

You can add both lines to your `~/.bashrc` to run them automatically on login.

### If using Jupyter

Run this once while the environment is activated:

```bash
python -m ipykernel install --user --name=naturalistic
```

---

## Docker

### One time: build the image

From inside the project folder:

```bash
docker build -t naturalistic .
```

### Every time to run a script

```bash
docker run --rm -v $(pwd):/app naturalistic python my_script.py
```

You could also use an alias to make the command shorter, like:

```bash
echo "alias drun='docker run --rm -v \$(pwd):/app naturalistic'" >> ~/.bashrc
source ~/.bashrc
drun python my_script.py
```
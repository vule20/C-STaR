##
```bash
cd STaRC/
conda env create -f environment.yaml
```

Note by Vu: For the llama env, request a node with one gpu and install dependencies with this gpu node as follows:
```
salloc --gres=gpu:a100:1 --partition=gpu-preempt --nodes=1 --ntasks=1 --cpus-per-gpu=16
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6
conda create --name llama3 python=3.9
pip install -r requirements_llama3.txt --no-cache
```
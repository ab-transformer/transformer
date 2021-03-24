import os

for sample in [10, 50, 100, 200, 300, 400, None]:
    command = f"python train_all.py --vonly --project_name vonly_sample --batch_size 32 --lr 0.0001  --l_sample {sample}"
    os.system(command)

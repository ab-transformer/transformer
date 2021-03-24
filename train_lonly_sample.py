import os

for sample in [10, 20, 30, 40, 50, None]:
    command = f"python train_all.py --lonly --project_name lonly_sample --batch_size 128 --lr 0.0001  --l_sample {sample}"
    os.system(command)

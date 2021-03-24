import os

for sample in [10, 50, 100, 500, 1000, None]:
    command = f"python train_all.py --aonly --project_name aonly_sample --batch_size 8 --lr 0.0001  --l_sample {sample}"
    os.system(command)

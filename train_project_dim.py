import os

for project_dim in [256]:  # [32, 64, 128, 256]:
    command = f"python train.py --project_name project_dim --batch_size 8 --lr 0.0001 --shuffle --project_dim {project_dim}"
    os.system(command)

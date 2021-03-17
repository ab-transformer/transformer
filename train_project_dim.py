import os

for project_dim in [60, 120, 240]:
    command = f"python train.py --project_name project_dim --batch_size 4 --lr 0.0001 --shuffle --project_dim {project_dim}"
    os.system(command)

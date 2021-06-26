import subprocess
from copy import copy

from ray import tune
from ray.tune.suggest.basic_variant import BasicVariantGenerator


def train(c):
    subprocess.run(
        f"python train_all.py --project_name report_mosei --lr 0.001 --shuffle --project_dim 32 --layers 4 --num_heads 8 --dataset mosei_sent --batch_size 16 --relu_dropout {c['relu_dropout']} --res_dropout {c['res_dropout']} --embed_dropout {c['embed_dropout']} --attn_dropout {c['attn_dropout']} --attn_dropout_a {c['attn_dropout_a']} --attn_dropout_v {c['attn_dropout_v']} --out_dropout {c['out_dropout']} --num_epochs 20 --norm".split(),
        cwd="/workspace/transformer",
    )


org_config = {
    "attn_dropout": 0.0,
    "attn_dropout_a": 0.1,
    "attn_dropout_v": 0.1,
    "embed_dropout": 0.1,
    "out_dropout": 0.1,
    "relu_dropout": 0.1,
    "res_dropout": 0.1,
}

points_to_evaluate = []
for k, v in org_config.items():
    point = copy(org_config)
    if point[k] + 0.1 <= 1.0:
        point[k] += 0.1
        points_to_evaluate.append(point)
    if point[k] - 0.1 > 0.0:
        point = copy(org_config)
        point[k] -= 0.1
    points_to_evaluate.append(point)

search_space = {k: tune.quniform(0, 0.5, 0.05) for k in org_config.keys()}

# for p in points_to_evaluate:
#     print(p)

tune.run(
    train,
    config=search_space,
    resources_per_trial={"cpu": 16, "gpu": 1},
    search_alg=BasicVariantGenerator(points_to_evaluate=points_to_evaluate),
    name="tune_mosei_sent_dropouts",
    local_dir="ray_results",
)

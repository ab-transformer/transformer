import argparse

from pytorch_lightning.loggers import CSVLogger, CometLogger
import pytorch_lightning as pl
import torch as th

from lightningmodule import MULTModelWarped
from datasets import load_impressionv2_dataset_all

parser = argparse.ArgumentParser()
parser.add_argument("-f", default="", type=str)

# Fixed
parser.add_argument(
    "--model",
    type=str,
    default="MulT",
    help="name of the model to use (Transformer, etc.)",
)

# Tasks
parser.add_argument(
    "--vonly",
    action="store_true",
    help="use the crossmodal fusion into v (default: False)",
)
parser.add_argument(
    "--aonly",
    action="store_true",
    help="use the crossmodal fusion into a (default: False)",
)
parser.add_argument(
    "--lonly",
    action="store_true",
    help="use the crossmodal fusion into l (default: False)",
)
# parser.add_argument('--aligned', action='store_true',
#                     help='consider aligned experiment or not (default: False)')
# parser.add_argument('--dataset', type=str, default='mosei_senti',
#                     help='dataset to use (default: mosei_senti)')
# parser.add_argument('--data_path', type=str, default='data',
#                     help='path for storing the dataset')

# Dropouts
parser.add_argument("--attn_dropout", type=float, default=0.1, help="attention dropout")
parser.add_argument(
    "--attn_dropout_a", type=float, default=0.0, help="attention dropout (for audio)"
)
parser.add_argument(
    "--attn_dropout_v", type=float, default=0.0, help="attention dropout (for visual)"
)
parser.add_argument("--relu_dropout", type=float, default=0.1, help="relu dropout")
parser.add_argument(
    "--embed_dropout", type=float, default=0.25, help="embedding dropout"
)
parser.add_argument(
    "--res_dropout", type=float, default=0.1, help="residual block dropout"
)
parser.add_argument(
    "--out_dropout", type=float, default=0.0, help="output layer dropout"
)

# Architecture
parser.add_argument(
    "--layers", type=int, default=5, help="number of layers in the network (default: 5)"
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=5,
    help="number of heads for the transformer network (default: 5)",
)
parser.add_argument(
    "--attn_mask",
    action="store_false",
    help="use attention mask for Transformer (default: true)",
)

# Tuning
parser.add_argument(
    "--batch_size", type=int, default=8, metavar="N", help="batch size (default: 8)"
)
# parser.add_argument('--clip', type=float, default=0.8,
#                     help='gradient clip value (default: 0.8)')
parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
# parser.add_argument('--optim', type=str, default='Adam',
#                     help='optimizer to use (default: Adam)')
parser.add_argument(
    "--num_epochs", type=int, default=1000, help="number of epochs"
)
# parser.add_argument('--when', type=int, default=20,
#                     help='when to decay learning rate (default: 20)')
# parser.add_argument('--batch_chunk', type=int, default=1,
#                     help='number of chunks per batch (default: 1)')

# Logistics
# parser.add_argument('--log_interval', type=int, default=30,
#                     help='frequency of result logging (default: 30)')
# parser.add_argument('--seed', type=int, default=1111,
#                     help='random seed')
# parser.add_argument('--no_cuda', action='store_true',
#                     help='do not use cuda')
parser.add_argument(
    "--name", type=str, default="mult", help='name of the trial (default: "mult")'
)
args = parser.parse_args()

valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

del args.f
hyp_params = args

[train_ds, valid_ds, test_ds], target_names = load_impressionv2_dataset_all()
train_dl = th.utils.data.DataLoader(
    train_ds, batch_size=hyp_params.batch_size, pin_memory=True,
)
valid_dl = th.utils.data.DataLoader(
    valid_ds, batch_size=hyp_params.batch_size, pin_memory=True,
)
test_dl = th.utils.data.DataLoader(
    test_ds, batch_size=hyp_params.batch_size, pin_memory=True,
)

audio, face, text, label = next(iter(train_dl))
hyp_params.orig_d_l = text.shape[2]
hyp_params.orig_d_a = audio.shape[2]
hyp_params.orig_d_v = face.shape[2]
hyp_params.l_len = text.shape[1]
hyp_params.a_len = audio.shape[1]
hyp_params.v_len = face.shape[1]
# hyp_params.use_cuda = True
# hyp_params.dataset = dataset
# hyp_params.when = args.when
# hyp_params.batch_chunk = args.batch_chunk
# hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
# hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = label.shape[1]  # output_dim_dict.get(dataset, 1)
# hyp_params.criterion = th.nn.L1Loss #criterion_dict.get(dataset, 'L1Loss')

model = MULTModelWarped(hyp_params, target_names)

if __name__ == "__main__":
    csv_logger = CSVLogger("logs", name="my_exp_name")
    comet_logger = CometLogger(
        api_key="cgss7piePhyFPXRw1J2uUEjkQ",
        workspace="transformer",
        project_name="find_lr",
    )
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=hyp_params.num_epochs,
        log_every_n_steps=1,
        logger=[csv_logger, comet_logger],
        overfit_batches=1
    )
    trainer.fit(model, train_dl, valid_dl)

    trainer.test(test_dataloaders=test_dl)

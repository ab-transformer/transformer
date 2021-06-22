import argparse
import random

import numpy as np
import torch as th

from datasets import load_impressionv2_dataset_all, load_resampled_impressionv2_dataset_all, load_report_impressionv2_dataset_all, load_report_mosi_dataset_all

parser = argparse.ArgumentParser()
parser.add_argument("-f", default="", type=str)

# Fixed
parser.add_argument(
    "--model",
    type=str,
    default="MulT",
    help="name of the model to use (Transformer, etc.)",
)

parser.add_argument(
    "--loss_fnc",
    type=str,
    default="L2",
    help="name of the loss function to use (default: Bell)",
)

# Tasks
parser.add_argument(
    "--dataset",
    type=str,
    default="impressionV2",
    help="name of the dataset (impressionV2, report)",
)
parser.add_argument(
    "--norm",
    action="store_true",
    help="normalize audio and visual modalities (default: False)",
)
parser.add_argument(
    "--resampled",
    action="store_true",
    help="use the resampled dataset (default: False)",
)
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
parser.add_argument(
    "--v_sample",
    type=int,
    default=None,
    help="vision modality samples, if none all time steps are used - Not available if resampled!",
)
parser.add_argument(
    "--a_sample",
    type=int,
    default=None,
    help="audio modality samples, if none all time steps are used - Not available if resampled!",
)
parser.add_argument(
    "--l_sample",
    type=int,
    default=None,
    help="language modality samples, if none all time steps are used - Not available if resampled!",
)
parser.add_argument(
    "--random_sample",
    action="store_true",
    help="take random time stamps instead of equally spaced ones - Not available if resampled!",
)


parser.add_argument(
    "--audio_emb", type=str, default="lld", help="audio embedding (default: lld) - If resampled only wav2vec2 is supported!"
)
parser.add_argument(
    "--face_emb",
    type=str,
    default="resnet18",
    help="face embedding (default: resnet18) - If resampled only ig65m is supported!",
)
parser.add_argument(
    "--text_emb", type=str, default="bert", help="text embedding (default: bert) - If resampled only bert is supported!"
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
parser.add_argument(
    "--project_dim",
    type=int,
    default=30,
    help="dimension of the projected embedding (default: 30)",
)

# Tuning
parser.add_argument(
    "--batch_size", type=int, default=4, metavar="N", help="batch size (default: 4)"
)
# parser.add_argument('--clip', type=float, default=0.8,
#                     help='gradient clip value (default: 0.8)')

parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay")
parser.add_argument(
    "--optim", type=str, default="Adam", help="optimizer to use (default: Adam)"
)
parser.add_argument("--num_epochs", type=int, default=1000, help="number of epochs")
parser.add_argument(
    "--limit", type=float, default=1.0, help="the procentage of data to be used"
)
parser.add_argument("--shuffle", action="store_true", help="reshuffle the batches")
# parser.add_argument('--when', type=int, default=20,
#                     help='when to decay learning rate (default: 20)')
# parser.add_argument('--batch_chunk', type=int, default=1,
#                     help='number of chunks per batch (default: 1)')

# Logistics
# parser.add_argument('--log_interval', type=int, default=30,
#                     help='frequency of result logging (default: 30)')
parser.add_argument("--seed", type=int, default=-1, help="random seed")
# parser.add_argument('--no_cuda', action='store_true',
#                     help='do not use cuda')
parser.add_argument("--project_name", type=str, help="Project name")
args = parser.parse_args()

if args.seed == -1:
    args.seed = random.randint(0, 10000)
random.seed(args.seed)
np.random.seed(args.seed)
th.manual_seed(args.seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

del args.f
hyp_params = args

if hyp_params.dataset == "impressionV2":
    if args.resampled:
        [train_ds, valid_ds, test_ds], target_names = load_resampled_impressionv2_dataset_all()
    else:
        [train_ds, valid_ds, test_ds], target_names = load_impressionv2_dataset_all(
            args.a_sample,
            args.v_sample,
            args.l_sample,
            args.random_sample,
            args.audio_emb,
            args.face_emb,
            args.text_emb,
        )
elif hyp_params.dataset == "report":
    train_ds, valid_ds, test_ds = load_report_impressionv2_dataset_all(args.norm)
elif hyp_params.dataset == "mosi":
    train_ds, valid_ds, test_ds = load_report_mosi_dataset_all(args.norm)
else:
    raise "Dataset not supported!"

# audio, face, text, label = next(iter(train_dl))
audio, face, text, label = train_ds[0]
hyp_params.orig_d_l = text.shape[1]
hyp_params.orig_d_a = audio.shape[1]
hyp_params.orig_d_v = face.shape[1]
hyp_params.l_len = text.shape[0]
hyp_params.a_len = audio.shape[0]
hyp_params.v_len = face.shape[0]
# hyp_params.use_cuda = True
# hyp_params.dataset = dataset
# hyp_params.when = args.when
# hyp_params.batch_chunk = args.batch_chunk
# hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
# hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = label.shape[0]  # output_dim_dict.get(dataset, 1)
# hyp_params.criterion = th.nn.L1Loss #criterion_dict.get(dataset, 'L1Loss')

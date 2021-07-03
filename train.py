import argparse
import itertools

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yaml
from bisect import bisect

from architecture.data.dataset import VisDialDataset
from architecture.encoders import Encoder
from architecture.decoders import Decoder
from architecture.metrics import SparseGTMetrics, NDCG
from architecture.model import EncoderDecoderModel
from architecture.utils.checkpointing import CheckpointManager, load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml",
    default="configs/lf_disc_faster_rcnn_x101.yml",
    help="Path to a config file listing reader, model and solver parameters.",
)
parser.add_argument(
    "--train-json",
    default="data/visdial_1.0_train.json",
    help="Path to json file containing VisDial v1.0 training data.",
)
parser.add_argument(
    "--val-json",
    default="data/visdial_1.0_val.json",
    help="Path to json file containing VisDial v1.0 validation data.",
)
parser.add_argument(
    "--val-dense-json",
    default="data/visdial_1.0_val_dense_annotations.json",
    help="Path to json file containing VisDial v1.0 validation dense ground "
    "truth annotations.",
)
parser.add_argument(
    "--captions-train-json",
    default="data/train2018.json",
    help="Path to json file containing VisDial v1.0 training captions data.",
)
parser.add_argument(
    "--captions-val-json",
    default="data/val2018.json",
    help="Path to json file containing VisDial v1.0 validation captions data.",
)
parser.add_argument(
    "--subset",
    type=int,
    default=120000,
    help="Number of training instances. (default is 120000)",
)


parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument(
    "--gpu-ids", nargs="+", type=int, default=0, help="List of ids of GPUs to use.",
)
parser.add_argument(
    "--cpu-workers", type=int, default=4, help="Number of CPU workers for dataloader.",
)
parser.add_argument(
    "--overfit",
    action="store_true",
    help="Overfit model on 5 examples, meant for debugging.",
)
parser.add_argument(
    "--validate",
    action="store_true",
    help="Whether to validate on val split after every epoch.",
)
parser.add_argument(
    "--in-memory",
    action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. "
    "Use only in presence of large RAM, atleast few tens of GBs.",
)


parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--save-dirpath",
    default="checkpoints/",
    help="Path of directory to create checkpoint directory and save " "checkpoints.",
)
parser.add_argument(
    "--load-pthpath",
    default="",
    help="To continue training, path to .pth file of saved checkpoint.",
)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# -------------------------------------------------------------------------
#   INPUT ARGUMENTS AND CONFIG
# -------------------------------------------------------------------------

args = parser.parse_args()

# keys: {"dataset", "model", "solver"}
config = yaml.load(open(args.config_yml))

if isinstance(args.gpu_ids, int):
    args.gpu_ids = [args.gpu_ids]
device = (
    torch.device("cuda", args.gpu_ids[0])
    if args.gpu_ids[0] >= 0
    else torch.device("cpu")
)

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))


# -------------------------------------------------------------------------
#   SETUP DATASET, DATALOADER, MODEL, CRITERION, OPTIMIZER, SCHEDULER
# -------------------------------------------------------------------------

train_dataset = VisDialDataset(
    config["dataset"],
    args.train_json,
    args.captions_train_json,
    overfit=args.overfit,
    in_memory=args.in_memory,
    num_workers=args.cpu_workers,
    return_options=True if config["model"]["decoder"] == "disc" else False,
    add_boundary_toks=False if config["model"]["decoder"] == "disc" else True,
)
train_subset = Subset(train_dataset, np.arange(args.subset))
train_dataloader = DataLoader(
    train_subset,
    batch_size=config["solver"]["batch_size"],
    num_workers=args.cpu_workers,
    shuffle=True,
)

val_dataset = VisDialDataset(
    config["dataset"],
    args.val_json,
    args.captions_val_json,
    args.val_dense_json,
    overfit=args.overfit,
    in_memory=args.in_memory,
    num_workers=args.cpu_workers,
    return_options=True,
    add_boundary_toks=False if config["model"]["decoder"] == "disc" else True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config["solver"]["batch_size"]
    if config["model"]["decoder"] == "disc"
    else 5,
    num_workers=args.cpu_workers,
)

# Pass vocabulary to construct Embedding layer.
encoder = Encoder(config["model"], train_dataset.vocabulary)
decoder = Decoder(config["model"], train_dataset.vocabulary)
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

# Share word embedding between encoder and decoder.
decoder.word_embed = encoder.word_embed

# Wrap encoder and decoder in a model.
model = EncoderDecoderModel(encoder, decoder).to(device)
if -1 not in args.gpu_ids:
    model = nn.DataParallel(model, args.gpu_ids)

# Loss function.
if config["model"]["decoder"] == "disc":
    criterion = nn.CrossEntropyLoss()
elif config["model"]["decoder"] == "gen":
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocabulary.PAD_INDEX)
else:
    raise NotImplementedError

if config["solver"]["training_splits"] == "trainval":
    iterations = (len(train_dataset) + len(val_dataset)) // config["solver"][
        "batch_size"
    ] + 1
else:
    iterations = len(train_dataset) // config["solver"]["batch_size"] + 1


def lr_lambda_fun(current_iteration: int) -> float:
    """Returns a learning rate multiplier.

    Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
    and then gets multiplied by `lr_gamma` every time a milestone is crossed.
    """
    current_epoch = float(current_iteration) / iterations
    if current_epoch <= config["solver"]["warmup_epochs"]:
        alpha = current_epoch / float(config["solver"]["warmup_epochs"])
        return config["solver"]["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(config["solver"]["lr_milestones"], current_epoch)
        return pow(config["solver"]["lr_gamma"], idx)


optimizer = optim.Adamax(model.parameters(), lr=config["solver"]["initial_lr"])
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fun)
T = iterations * (
    config["solver"]["num_epochs"] - config["solver"]["warmup_epochs"] + 1
)
scheduler2 = lr_scheduler.CosineAnnealingLR(
    optimizer, int(T), eta_min=config["solver"]["eta_min"], last_epoch=-1
)

# -------------------------------------------------------------------------
#   SETUP BEFORE TRAINING LOOP
# -------------------------------------------------------------------------

checkpoint_manager = CheckpointManager(
    model, optimizer, args.save_dirpath, config=config
)
sparse_metrics = SparseGTMetrics()
ndcg = NDCG()

# If loading from checkpoint, adjust start epoch and load parameters.
if args.load_pthpath == "":
    start_epoch = 0
else:
    # "path/to/checkpoint_xx.pth" -> xx
    start_epoch = int(args.load_pthpath.split("_")[-1][:-4]) + 1

    model_state_dict, optimizer_state_dict = load_checkpoint(args.load_pthpath)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    print("Loaded model from {}".format(args.load_pthpath))

# -------------------------------------------------------------------------
#   TRAINING LOOP
# -------------------------------------------------------------------------

# Forever increasing counter to keep track of iterations (for tensorboard log).
global_iteration_step = start_epoch * iterations

for epoch in range(start_epoch, config["solver"]["num_epochs"]):

    # -------------------------------------------------------------------------
    #   ON EPOCH START  (combine dataloaders if training on train + val)
    # -------------------------------------------------------------------------
    if config["solver"]["training_splits"] == "trainval":
        combined_dataloader = itertools.chain(train_dataloader, val_dataloader)
    else:
        combined_dataloader = itertools.chain(train_dataloader)

    print(f"\nTraining for epoch {epoch}:")
    for i, batch in enumerate(tqdm(combined_dataloader)):
        for key in batch:
            batch[key] = batch[key].to(device)

        optimizer.zero_grad()
        output = model(batch)
        target = (
            batch["ans_ind"]
            if config["model"]["decoder"] == "disc"
            else batch["ans_out"]
        )
        batch_loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
        batch_loss.backward()
        optimizer.step()

        if global_iteration_step <= iterations * config["solver"]["warmup_epochs"]:
            scheduler.step(global_iteration_step)
        else:
            global_iteration_step_in_2 = (
                iterations * config["solver"]["warmup_epochs"]
                + 1
                - global_iteration_step
            )
            scheduler2.step(int(global_iteration_step_in_2))
        global_iteration_step += 1
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    #   ON EPOCH END  (checkpointing and validation)
    # -------------------------------------------------------------------------
    checkpoint_manager.step()

    # Validate and report automatic metrics.
    if args.validate:

        # Switch dropout, batchnorm etc to the correct mode.
        model.eval()

        print(f"\nValidation after epoch {epoch}:")
        for i, batch in enumerate(tqdm(val_dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)
            with torch.no_grad():
                output = model(batch)
            sparse_metrics.observe(output, batch["ans_ind"])
            if "gt_relevance" in batch:
                output = output[torch.arange(output.size(0)), batch["round_id"] - 1, :]
                ndcg.observe(output, batch["gt_relevance"])

        all_metrics = {}
        all_metrics.update(sparse_metrics.retrieve(reset=True))
        all_metrics.update(ndcg.retrieve(reset=True))
        for metric_name, metric_value in all_metrics.items():
            print(f"{metric_name}: {metric_value}")
        # log the metrics
        with open(args.save_dirpath + "metrics_visdial.txt", "a+") as f:
            print(all_metrics, file=f)

        model.train()
        torch.cuda.empty_cache()

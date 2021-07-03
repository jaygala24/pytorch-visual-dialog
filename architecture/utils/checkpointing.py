from pathlib import Path
from subprocess import PIPE, Popen
import warnings

import torch
from torch import nn, optim
import yaml


class CheckpointManager(object):
    """A checkpoint manager saves state dicts of model and optimizer
    as .pth files in a specified directory. This class closely follows
    the API of PyTorch optimizers and learning rate schedulers.
    """

    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dirpath,
        step_size=1,
        last_epoch=-1,
        **kwargs,
    ):

        if not isinstance(model, nn.Module):
            raise TypeError("{} is not a Module".format(type(model).__name__))

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))

        self.model = model
        self.optimizer = optimizer
        self.ckpt_dirpath = Path(checkpoint_dirpath)
        self.step_size = step_size
        self.last_epoch = last_epoch
        self.init_directory(**kwargs)

    def init_directory(self, config={}):
        """Initialize empty checkpoint directory and save hyper-parameters config 
        in this directory to associate checkpoints with their hyper-parameters.
        """

        self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)
        yaml.dump(
            config,
            open(str(self.ckpt_dirpath / "config.yml"), "w"),
            default_flow_style=False,
        )

    def step(self, epoch=None):
        """Save checkpoint if step size conditions meet."""

        if not epoch:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if not self.last_epoch % self.step_size:
            torch.save(
                {
                    "model": self._model_state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                self.ckpt_dirpath / f"checkpoint_{self.last_epoch}.pth",
                _use_new_zipfile_serialization=False,
            )

    def _model_state_dict(self):
        """Returns state dict of model, taking care of DataParallel case."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()


def load_checkpoint(checkpoint_pthpath):
    """Given a path to saved checkpoint, load corresponding state dicts
    of model and optimizer from it.
    """

    # load encoder, decoder, optimizer state_dicts
    components = torch.load(checkpoint_pthpath)
    return components["model"], components["optimizer"]

import sys
import torch
import pytorch_lightning as pl
import socket
import os
import logging
import contextlib

if "mac" in socket.gethostname():
    sys.path.append(
        "/Users/sidharrthnagappan/Documents/University/Cambridge/Courses/Dissertation/dissertation/src"
    )
else:
    sys.path.append("/home/sn666/dissertation/src")

from models.hist_to_transcriptomics import HistopathologyToTranscriptomics


@contextlib.contextmanager
def suppress_logging():
    # Disable all logging messages of level CRITICAL and below.
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        # Re-enable logging.
        logging.disable(logging.NOTSET)


# TRANSCRIPTOMICS_MODEL_PATH = "/auto/archive/tcga/sn666/trained_models/hist_to_transcriptomics/h_to_t_uni_128_b_subsetgene_then_norm_relu/epoch=9-step=3950.ckpt"

TRANSCRIPTOMICS_MODEL_PATH = "/auto/archive/tcga/sn666/trained_models/hist_to_transcriptomics/h_to_t_uni_porpoise_genes_2layer/epoch=7-step=3160.ckpt"


class MiniPatchDataset(torch.utils.data.Dataset):
    """
    Brutally simple dataset to store individual patches coming in
    and prepare for inference on the HistopathologyToTranscriptomics model.
    """

    def __init__(self, patch_features: torch.Tensor):
        self.patch_features = patch_features

    def __len__(self):
        return self.patch_features.shape[0]

    def __getitem__(self, idx):
        # return a single patch's features
        # the existence of the foundation_model_features key tells the model to
        # skip the foundation model's inference during the full forward pass
        return {"foundation_model_features": self.patch_features[idx]}


def load_model(checkpoint_path: str):
    """
    Load the HistopathologyToTranscriptomics model from a checkpoint.
    """
    model = HistopathologyToTranscriptomics.load_from_checkpoint(checkpoint_path)
    return model


histtost = load_model(TRANSCRIPTOMICS_MODEL_PATH)


def get_transcriptomics_data(patch_features: torch.Tensor):
    """
    For each patch, return the transcriptomics data by calling a saved HistopathologyToTranscriptomics model.
    """
    # getting too noisy
    with suppress_logging():
        # there can be a variable number of patches passed to this function
        num_patches = patch_features.shape[0]
        # print(f"num_patches: {num_patches}")

        trainer_args = {
            "logger": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
        }
        dataset = MiniPatchDataset(patch_features)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_patches)
        if torch.cuda.is_available():
            trainer = pl.Trainer(devices=1, **trainer_args)
        else:
            trainer = pl.Trainer(accelerator="cpu", **trainer_args)

        # print("Predicting transcriptomics data...")
        predictions = trainer.predict(histtost, dataloaders=dataloader)
        # print("Predictions done.")

        return predictions


def get_num_transcriptomics_features():
    # TODO: make this dynamic idk
    return histtost.num_outputs


def load(path):
    # root_dir = '/home/sn666/rds/rds-cl-acs-qRKC0ovsKR0/sn666/healnet/data/tcga/tcga/wsi/luad_zzb20_uni'
    # assert root_dir is not None, f"set_preprocess_dir must be called before load!"
    # path = os.path.join(root_dir, slide_id + f"_{power:.3f}.pt")
    # assert os.path.isfile(path), f"Pre-process load: path '{path}' not found!"
    return torch.load(path)


if __name__ == "__main__":
    # load the patch features
    patch_features = load(
        "/home/sn666/rds/rds-cl-acs-qRKC0ovsKR0/sn666/healnet/data/tcga/tcga/wsi/luad_zzb20_uni/TCGA-4B-A93V-01Z-00-DX1.C263DC1C-298D-47ED-AAF8-128043828530_5.000.pt"
    )
    print(patch_features[0][0])
    transcriptomics_data = get_transcriptomics_data(patch_features[0][0])
    print(transcriptomics_data)
    # get the transcriptomics data

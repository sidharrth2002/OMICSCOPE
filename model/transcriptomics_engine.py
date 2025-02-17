import sys
import torch
import pytorch_lightning as pl
import socket

if "mac" in socket.gethostname():
    sys.path.append(
        "/Users/sidharrthnagappan/Documents/University/Cambridge/Courses/Dissertation/dissertation/src"
    )
else:
    sys.path.append(
        "/home/sn666/dissertation/src"
    )


from models.hist_to_transcriptomics import HistopathologyToTranscriptomics


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


histtost = load_model("model.ckpt")


def get_transcriptomics_data(patch_features: torch.Tensor):
    """
    For each patch, return the transcriptomics data by calling a saved HistopathologyToTranscriptomics model.
    """
    # there can be a variable number of patches passed to this function
    num_patches = patch_features.shape[0]
    dataset = MiniPatchDataset(patch_features)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_patches)
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0)

    predictions = trainer.predict(histtost, dataloaders=dataloader)

    return predictions

if __name__ == '__main__':
    
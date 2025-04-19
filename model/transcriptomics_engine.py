import sys
import torch
import pytorch_lightning as pl
import socket
import os
import logging
import contextlib
import hashlib      # stdlib is fine, but xxhash / blake3 are 5‑10× faster

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

TRANSCRIPTOMICS_MODEL_PATH = "/auto/archive/tcga/sn666/trained_models/hist_to_transcriptomics/h_to_t_uni_porpoise_genes_2layer/epoch=9-step=3950.ckpt"


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

"""
This global variable is used to store the transcriptomics observations of previously 
computed patches, so that we don't have to recompute them.

Why do this?
Because the same "high-resolution" patch can end up at multiple magnifications, and we don't want to recompute
"""
CACHE: dict[str, torch.Tensor] = {}     # fp16 preds on CPU to save RAM


def tensor_fingerprint(
    t: torch.Tensor,
    *,                         # force kw‑args
    ndigits: int | None = None # round to this many decimal places first
) -> str:
    """
    Return a hex string that is identical *iff* the tensor contents
    (after optional rounding) are identical.
    """
    if ndigits is not None:
        t = torch.round(t, decimals=ndigits)

    # Make sure we are hashing a contiguous CPU view with a stable dtype.
    # .contiguous() is a no‑op if the tensor is already C‑contiguous.
    buf = t.detach().to(dtype=torch.float32, device="cpu", copy=False).contiguous().numpy().tobytes()
    return hashlib.blake2b(buf, digest_size=8).hexdigest()  # 8 bytes → 16‑char hex

def get_transcriptomics_data(patch_features: torch.Tensor) -> torch.Tensor:
    B, P, _ = patch_features.shape
    device  = patch_features.device
    out_dim = histtost.num_outputs
    result  = torch.empty((B, P, out_dim), device=device)

    # -----------------------------------------------------------------------
    # 1) split cached vs. missing by hashing each slide tensor once
    # -----------------------------------------------------------------------
    missing_idx, fingerprints = [], []

    for i in range(B):
        fp = tensor_fingerprint(patch_features[i], ndigits=4)  # or None for exact
        if fp in CACHE:                      # hit
            result[i] = CACHE[fp].to(device)
        else:                                # miss
            fingerprints.append(fp)
            missing_idx.append(i)

    # -----------------------------------------------------------------------
    # 2) run the model once for all misses
    # -----------------------------------------------------------------------
    if missing_idx:
        feats  = patch_features[missing_idx]                 # (n_missing, P, 1024)
        loader = torch.utils.data.DataLoader(
            MiniPatchDataset(feats), batch_size=len(missing_idx)
        )

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False, enable_progress_bar=False, enable_model_summary=False,
        )

        with suppress_logging():
            preds = torch.cat(trainer.predict(histtost, dataloaders=loader), dim=0)  # (n_missing, P, out_dim)

        # write to output and cache
        result[missing_idx] = preds
        for fp, pred in zip(fingerprints, preds):
            CACHE[fp] = pred.to(dtype=torch.float16, device="cpu")  # light‑weight cache

    return result

# def get_transcriptomics_data(
#     patch_features: torch.Tensor,                       # (B, P, 1024)
#     identifiers: list[str],                             # len == B
# ) -> torch.Tensor:                                      # (B, P, num_outputs)
#     """
#     For every slide (B) consisting of P patch‑features (1024‑D),
#     return the transcriptomics predictions of `histtost`.
#     Results are cached in `COMPUTED_TRANSCRIPTOMICS[identifier]`
#     to avoid recomputation on later calls.
#     """
#     print(f"identifiers length: {len(identifiers)}")
#     B, P, _ = patch_features.shape
#     device         = patch_features.device
#     out_dim        = histtost.num_outputs
#     result         = torch.empty((B, P, out_dim), device=device)

#     # -------- 1. split cached vs. missing -----------------------------------
#     cached_idx, cached_tensors = [], []
#     missing_idx, missing_id    = [], []

#     for idx, ident in enumerate(identifiers):
#         if ident in COMPUTED_TRANSCRIPTOMICS:          # cached
#             cached_idx.append(idx)
#             cached_tensors.append(COMPUTED_TRANSCRIPTOMICS[ident].to(device))
#         else:                                          # still missing
#             missing_idx.append(idx)
#             missing_id.append(ident)

#     # fill the slides we already have
#     if cached_idx:
#         result[cached_idx] = torch.stack(cached_tensors, dim=0)   # (n_cached, P, out_dim)

#     # -------- 2. run the model once for the rest -----------------------------
#     if missing_idx:
#         feats  = patch_features[missing_idx]                        # (n_missing, P, 1024)
#         loader = torch.utils.data.DataLoader(
#             MiniPatchDataset(feats), batch_size=len(missing_idx)
#         )

#         trainer = pl.Trainer(
#             accelerator="gpu" if torch.cuda.is_available() else "cpu",
#             devices=1,
#             logger=False, enable_progress_bar=False, enable_model_summary=False,
#         )

#         with suppress_logging():
#             preds = torch.cat(trainer.predict(histtost, dataloaders=loader), dim=0)  # (n_missing, P, out_dim)

#         # write predictions into the output tensor and the global cache
#         result[missing_idx] = preds
#         for idx, ident, pred in zip(missing_idx, missing_id, preds):
#             print(f"pred shape: {pred.shape} before adding to cache")
#             COMPUTED_TRANSCRIPTOMICS[ident] = pred.detach().cpu()

#     print(f"result shape: {result.shape}")
#     return result

# def get_transcriptomics_data(patch_features: torch.Tensor, identifiers: list):
#     """
#     For each patch, return the transcriptomics data by calling a saved HistopathologyToTranscriptomics model.
#     """
#     # first check if the patch features are already computed
#     # if so, return the cached value
#     print(f"patch features shape: {patch_features.shape}")
    
#     return_value = torch.zeros(
#         (patch_features.shape[0], patch_features.shape[1], histtost.num_outputs), device=patch_features.device
#     )
#     num_patches = patch_features.shape[0]
#     for i in range(patch_features.shape[0]):
#         if identifiers[i] in COMPUTED_TRANSCRIPTOMICS:
#             # print(f"Using cached transcriptomics for {identifiers[i]}")
#             return_value[i][
    
#     # identify indicies that need to be computed
#     missing_inds = []
#     missing_features = []
#     missing_identifiers = []
#     for i in range(num_patches):
#         if identifiers[i] not in COMPUTED_TRANSCRIPTOMICS:
#             missing_inds.append(i)
#             missing_features.append(patch_features[i])
#             missing_identifiers.append(identifiers[i])
    
#     if missing_inds:
#         missing_features = torch.stack(missing_features, dim=0)
#         print(f"missing features shape: {missing_features.shape}")
#         dataset = MiniPatchDataset(missing_features)
#         dataloader = torch.utils.data.DataLoader(dataset, batch_size=missing_features.shape[0])
        
#         trainer_args = {
#             "logger": False,
#             "enable_progress_bar": False,
#             "enable_model_summary": False,
#         }
        
#         if torch.cuda.is_available():
#             trainer = pl.Trainer(devices=1, **trainer_args)
#         else:
#             trainer = pl.Trainer(accelerator="cpu", **trainer_args)
        
#         with suppress_logging():
#             predictions = trainer.predict(histtost, dataloaders=dataloader)
        
#         predictions = torch.cat(predictions, dim=0)
#         print(f"predictions shape: {predictions.shape}")
#         print(f"missing identifiers length: {len(missing_identifiers)}")
        
#         for idx, pred, ident in zip(missing_inds, predictions, missing_identifiers):
#             print(f"pred in loop: {pred.shape}")
#             COMPUTED_TRANSCRIPTOMICS[ident] = pred
#             return_value[idx] = pred
        
#     return return_value

    # # if not, compute the transcriptomics data
    # # getting too noisy
    # with suppress_logging():
    #     # there can be a variable number of patches passed to this function
    #     num_patches = patch_features.shape[0]
    #     # print(f"num_patches: {num_patches}")

    #     trainer_args = {
    #         "logger": False,
    #         "enable_progress_bar": False,
    #         "enable_model_summary": False,
    #     }
    #     dataset = MiniPatchDataset(patch_features)
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=num_patches)
    #     if torch.cuda.is_available():
    #         trainer = pl.Trainer(devices=1, **trainer_args)
    #     else:
    #         trainer = pl.Trainer(accelerator="cpu", **trainer_args)

    #     # print("Predicting transcriptomics data...")
    #     predictions = trainer.predict(histtost, dataloaders=dataloader)
    #     # print("Predictions done.")

    #     # store each prediction in the global variable COMPUTED_TRANSCRIPTOMICS, where key is the identifier
    #     for i in range(num_patches):
    #         if identifiers[i] not in COMPUTED_TRANSCRIPTOMICS:
    #             print(f"Adding computed transcriptomics for {identifiers[i]} to cache")
    #             COMPUTED_TRANSCRIPTOMICS[identifiers[i]] = predictions[i]

    #     return predictions


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

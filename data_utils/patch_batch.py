"""
The PatchBatch class. A batch of data is quite a complicated object here, containing variable length padded lists of
patches along with their positions, global + local context vectors and hierarchical information. The PatchBatch class
presents a simple interface for this complicated collection of data.
"""

import torch
from typing import Tuple, Dict
import os
from model.transcriptomics_engine import get_num_transcriptomics_features, get_transcriptomics_data

from .slide import RawSlide, PreprocessedSlide
import utils


class PatchBatch:
    """
    A batch of single magnification slide data, used at both train and inference time. Contains sanity checks,
    computes the padding mask, and keeps the code clean (rather than passing around these ~7 arguments).

    Essentially handles some of the complexity of processing complex objects of different lengths in one batch.
    """

    def __init__(
        self,
        locs: torch.LongTensor,
        num_ims: torch.LongTensor,
        parent_inds: torch.LongTensor,
        ctx_slide: torch.Tensor,
        ctx_patch: torch.Tensor,
        fts: torch.Tensor,
        transcriptomics: torch.Tensor = None,
        **unused_kwargs
    ):
        """
        :param locs: locations of each patch feature, in *pixel coordinates at that magnification*. Shape (B x N x 2)
        :param num_ims: number of images in each batch. Shape (B)
        :param parent_inds: parent index (at prev magnification) of each patch. Useful for visualisation. shape (B x N).
        :param ctx_slide: slide-level context (i.e. F^1, F^2, ...). shape (B x Depth x D_slide)
        :param ctx_patch: patch-level hierarchical context. shape (B x N x Depth x D_patch).
          Note: when LSTM is used, this is actually used to store LSTM state rather than patch features.
        :param fts: patch features. Shape (B x N x D). Padded features are the zero vector, but should not be exposed
          to the model at any point.
        :param transcriptomics: transcriptomics features. Predicted gene expression data for each patch.
        """
        batch_size, max_patches, c = fts.shape

        _, self.ctx_depth, self.ctx_dim1 = ctx_slide.shape
        self.ctx_dim2 = ctx_patch.shape[-1]

        # Check all shapes
        assert locs.shape == (batch_size, max_patches, 2)
        assert num_ims.shape == (batch_size,)
        assert parent_inds.shape == (batch_size, max_patches)
        assert ctx_slide.shape == (batch_size, self.ctx_depth, self.ctx_dim1)
        assert ctx_patch.shape == (
            batch_size,
            max_patches,
            self.ctx_depth,
            self.ctx_dim2,
        )

        assert num_ims.max().item() == max_patches

        # Obtain and check device
        self.device = fts.device
        assert (
            self.device
            == locs.device
            == num_ims.device
            == parent_inds.device
            == ctx_slide.device
            == ctx_patch.device
            == fts.device
        )

        self.batch_size = batch_size
        self.max_patches = max_patches

        self.fts = fts
        self.transcriptomics = transcriptomics
        self.locs = locs
        self.num_ims = num_ims
        self.parent_inds = parent_inds
        self.ctx_slide = ctx_slide
        self.ctx_patch = ctx_patch

        # Create indices which are in range using num_ims
        inds = torch.arange(max_patches, device=num_ims.device).expand(batch_size, -1)
        inds = inds < num_ims[:, None]
        # Now self.patches[valid_inds] will extract only non-padding patches
        self.valid_inds = inds


def from_batch(batch: Dict, device, transcriptomics_type: str, transcriptomics_model_path: str) -> PatchBatch:
    transcriptomics_dim = get_num_transcriptomics_features(transcriptomics_model_path)
    
    if transcriptomics_type == 'multi-magnification':
        # store unique identifiers for each patch, made up of slide id and patch locs
        # get transcriptomics based on the patch features
        transcriptomics = get_transcriptomics_data(batch['fts'], transcriptomics_model_path)
        batch['transcriptomics'] = transcriptomics
    
    # TODO: Complete this!
    elif transcriptomics_type == 'highest-magnification':
        # we only want to do transcriptomics on the leaves, i.e. the highest magnification patches
        # shape is [batch size x max images x num patches x embedding dim]
        # since it's padded, I only want to do transcriptomics on the non-zero patches and average them

        leaf_fts   = batch["leaf_fts_grouped"]                 # (B, P_max, C_max, D)
        B, P_max, C_max, D = leaf_fts.shape

        # ------------------------------------------------------------------ #
        # 1)  flatten once → pick only the valid children                    #
        # ------------------------------------------------------------------ #
        mask           = leaf_fts.abs().sum(dim=-1) != 0       # (B, P, C)  True → real child
        valid_feats    = leaf_fts[mask]                        # (N_total, D)

        if valid_feats.numel() != 0:                           # rare edge‑case
            # indices to map every child back to (slide, patch)
            slide_idx = (
                torch.arange(B)[:, None, None]
                .expand(B, P_max, C_max)[mask]                     # (N_total,)
            )
            patch_idx = (
                torch.arange(P_max)[None, :, None]
                .expand(B, P_max, C_max)[mask]                     # (N_total,)
            )

            # ------------------------------------------------------------------ #
            # 2)  ***single*** transcriptomics inference for the whole batch     #
            # ------------------------------------------------------------------ #
            preds_child = get_transcriptomics_data(valid_feats, transcriptomics_model_path)
            # shape: (N_total, T)

            # ------------------------------------------------------------------ #
            # 3)  scatter‑add → sum over children, then divide by count          #
            # ------------------------------------------------------------------ #
            agg   = torch.zeros(B, P_max, transcriptomics_dim)       # sum of children
            count = torch.zeros(B, P_max, 1)        # #children per patch

            idx_flat = slide_idx * P_max + patch_idx               # linear index over (B, P_max)
            agg.view(-1, transcriptomics_dim).scatter_add_(0,
                idx_flat.unsqueeze(1).expand(-1, transcriptomics_dim), preds_child)

            count.view(-1, 1).scatter_add_(0,
                idx_flat.unsqueeze(1), torch.ones_like(idx_flat, dtype=torch.float32).unsqueeze(1))

            # average (safe – patches with 0 children stay 0)
            batch["transcriptomics"] = (agg / count.clamp(min=1.0))  # (B, P_max, T)
        else:
            # no children -> zero transcriptomics vector
            batch["transcriptomics"] = torch.zeros(B, P_max, transcriptomics_dim)


        # print(f"Elements in batch: {batch.keys()}")
        # print(f"Batch fts length: {len(batch['fts'])}")
        # print(f"Batch fts[0] shape: {batch['fts'][0].shape}")
        # if "leaf_fts_grouped" in batch:
        #     print(f"Batch leaf_fts_grouped length: {len(batch['leaf_fts_grouped'])}")
        #     print(f"Batch leaf_fts_grouped[0] shape: {batch['leaf_fts_grouped'][0].shape}")
        #     print(f"Batch leaf_fts_grouped[1] shape: {batch['leaf_fts_grouped'][1].shape}")
        
        # # leaf fts grouped: list of B tensors, each of shape [patches, children, embedding dim]
        # tx_list = []
        # for leaf_feats in batch['leaf_fts_grouped']:
        #     print(f"Leaf feats shape: {leaf_feats.shape}")
        #     P, C, D = leaf_feats.shape[0], leaf_feats.shape[1], leaf_feats.shape[2]
            
        #     mask = leaf_feats.abs().sum(dim=-1) != 0
            
        #     tx_per_patch = []
        #     for i in range(P):
        #         valid_feats = leaf_feats[i][mask[i]]
        #         if valid_feats.numel() == 0:
        #             # no children -> zero transcriptomics vector
        #             tx = torch.zeros(transcriptomics_dim, device=batch['fts'].device)
        #         else:
        #             # run model once on all Ni children
        #             tx_children = get_transcriptomics_data(valid_feats, transcriptomics_model_path)

        #             # average over children dimension
        #             tx = tx_children.mean(dim=0)
        #         tx_per_patch.append(tx)
            
        #     # stack to [P, T]
        #     tx_per_patch = torch.stack(tx_per_patch, dim=0)
        #     tx_list.append(tx_per_patch)
        
        # batch['transcriptomics'] = torch.stack(tx_list, dim=0)

    print("transcriptomics", batch['transcriptomics'].shape)

    batch = {i: utils.todevice(j, device) for i, j in batch.items()}
    return PatchBatch(**batch)


def from_raw_slide(slide: RawSlide, im_enc, transform, device=None) -> PatchBatch:
    """
    Creates a PatchBatch object from a RawSlide + Image Encoder, ready to be input to the model.

    Note: carries out image preprocessing, and patch loading if not loaded yet. All patches are encoded as a single
    batch, as we assume K is sufficiently low to allow this.
    """
    print('from_raw_slide function is running')
    if device is None:
        device = utils.device

    # Helper function: add singleton batch dim, move to cuda
    def p(x):
        if x is None:
            return x
        if isinstance(x, tuple) or isinstance(x, list):
            return [p(i) for i in x]
        return x[None].to(device)

    if slide.patches is None:
        slide.load_patches()
    with torch.no_grad():
        fts = im_enc(transform(slide.patches.to(device)))

    transcriptomics = get_transcriptomics_data(fts)

    num_ims = torch.LongTensor([slide.locs.size(0)]).to(device)
    return PatchBatch(
        p(slide.locs),
        num_ims,
        p(slide.parent_inds),
        p(slide.ctx_slide),
        p(slide.ctx_patch),
        p(fts),
        transcriptomics=p(transcriptomics),
    )


def from_preprocessed_slide(slide: PreprocessedSlide, device=None) -> PatchBatch:
    """
    Creates a PatchBatch object from a PreprocessedSlide. As the patches are preprocessed, there is no need for an image
    encoder.
    """
    if device is None:
        device = utils.device

    # Helper function: add singleton batch dim, move to cuda
    def p(x):
        if x is None:
            return x
        if isinstance(x, tuple) or isinstance(x, list):
            return [p(i) for i in x]
        return x[None].to(device)

    num_ims = torch.LongTensor([slide.locs.size(0)]).to(device)
    fts = slide.fts[0]
    # get transcriptomics data by predicting gene expression based on patch features
    transcriptomics = get_transcriptomics_data(fts)
    return PatchBatch(
        p(slide.locs),
        num_ims,
        p(slide.parent_inds),
        p(slide.ctx_slide),
        p(slide.ctx_patch),
        p(fts),
        transcriptomics=transcriptomics,
    )

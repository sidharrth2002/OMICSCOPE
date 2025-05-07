import time
import torch
from torch import nn
from typing import Dict, Tuple

from data_utils.patch_batch import PatchBatch
from .interface import Processor
from .aggregator import TransformerAggregator
import config as cfg
import utils
from model.transcriptomics_engine import get_num_transcriptomics_features
import torch.nn.functional as F
import wandb

class CombineTranscriptomicsPatchCtx(nn.Module):
    def __init__(self, feat_1_dim, feat_2_dim, hdim, dropout_p=0.1):
        super(CombineTranscriptomicsPatchCtx, self).__init__()
        # Define input and output dimensions
        input_dim = feat_1_dim + hdim + feat_2_dim
        output_dim = feat_1_dim + hdim

        # First transformation with layer normalization and dropout
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        self.dropout1 = nn.Dropout(p=dropout_p)

        # Second transformation with layer normalization and dropout
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.dropout2 = nn.Dropout(p=dropout_p)

        # Residual connection: if input dimension is not equal to output dimension, project it.
        self.residual_proj = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )

    def forward(self, feat_1, feat_2):
        x = torch.cat((feat_1, feat_2), dim=-1)
        # x = torch.cat((feat_1, feat_2.clone().detach()), dim=-1)
        print(f"feat_1 shape: {feat_1.shape}, feat_2 shape: {feat_2.shape}, x shape: {x.shape}")
        # x shape: [batch_size, input_dim]
        residual = self.residual_proj(x)

        # First layer transformation
        out = self.fc1(x)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        # Second layer transformation
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.dropout2(out)

        # Residual addition with activation
        out = F.relu(out + residual)
        return out

class GatedTranscriptomicsFusion(nn.Module):
    def __init__(self, feat1_dim, feat2_dim, hidden_dim, dropout_p=0.1):
        super().__init__()
        self.enricher = CombineTranscriptomicsPatchCtx(
            feat_1_dim=feat1_dim,
            feat_2_dim=feat2_dim,
            hdim=hidden_dim,
            dropout_p=dropout_p
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(feat1_dim + hidden_dim + feat2_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(feat1_dim + hidden_dim + feat2_dim)

    def forward(self, feat_1, feat_2):
        # feat_1: [B,N,D], feat_2: [B,N,F]
        # 1) mask: which patches actually have transcriptomics
        valid = (feat_2.abs().sum(dim=-1, keepdim=True) != 0).float()  # [B,N,1]

        # 2) compute gate normally, then zero it where invalid
        # TODO: return back to normal if this becomes a problem
        g = self.gate_mlp(self.norm(torch.cat([feat_1, feat_2], dim=-1)))  # [B,N,1]
        # g = self.gate_mlp(torch.cat([feat_1, feat_2], dim=-1))        # [B,N,1]
        
        # TODO: put this back
        g = g * valid                                                 # force g=0 for no-data

        # 3) enrichment
        enriched = self.enricher(feat_1, feat_2)                      # [B,N,D]

        wandb.log({
            "gated_enrichment": g.mean(),
            "g_hist": wandb.Histogram(g.detach().cpu().numpy())
        })

        # 4) fuse
        return g * enriched + (1.0 - g) * feat_1                      # [B,N,D]

class AttentionWeightedSum(nn.Module):
    def __init__(self, feat_1_dim, feat_2_dim):
        super(AttentionWeightedSum, self).__init__()
        # TODO: finish this, refer here: https://github.com/sidharrth2002/text-scoring/blob/main/implementations/model/tabular_combiner.py
        pass

class PATHSProcessor(nn.Module, Processor):
    """
    Implementation of a processor for one magnification level, Pi, implementing the abstract class Processor.
    The full model is an `interface.RecursiveModel` containing several `PATHSProcessor`s.
    """

    def __init__(self, config, train_config, depth: int):
        super().__init__()
        train_config: cfg.Config
        config: cfg.PATHSProcessorConfig

        self.depth = depth

        # Output dimensionality
        num_logits = train_config.num_logits()

        self.config = config
        self.train_config = train_config

        if config.model_dim is None:
            self.proj_in = nn.Identity()
            self.dim = config.patch_embed_dim
        else:
            self.proj_in = nn.Linear(
                config.patch_embed_dim, config.model_dim, bias=False
            )
            self.dim = config.model_dim

        self.slide_ctx_dim = config.trans_dim

        # Slide context can either be concatenated or summed; in our paper we choose sum (mode="residual")
        if self.config.slide_ctx_mode == "concat":
            self.classification_layer = nn.Linear(
                self.slide_ctx_dim * (depth + 1), num_logits
            )
        else:
            self.classification_layer = nn.Linear(self.slide_ctx_dim, num_logits)

        # Per-patch MLP to produce importance values \alpha
        # if self.config.add_transcriptomics:
        #     self.importance_mlp = nn.Sequential(
        #         nn.Linear(self.dim + get_num_transcriptomics_features(), config.importance_mlp_hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(config.importance_mlp_hidden_dim, 1),
        #     )
        # else:
        self.importance_mlp = nn.Sequential(
            nn.Linear(self.dim, config.importance_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.importance_mlp_hidden_dim, 1),
        )

        if config.lstm:
            self.hdim = config.hierarchical_ctx_mlp_hidden_dim
        else:
            # A simple RNN instead of the LSTM
            self.hctx_mlp = nn.Sequential(
                nn.Linear(self.dim, config.hierarchical_ctx_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hierarchical_ctx_mlp_hidden_dim, self.dim),
            )

        # Global aggregator
        self.global_agg = TransformerAggregator(
            input_dim=self.dim,
            model_dim=config.trans_dim,
            output_dim=self.dim,
            nhead=config.trans_heads,
            layers=config.trans_layers,
            dropout=config.dropout,
        )

        # self.transcriptomics_projector = nn.Linear(get_num_transcriptomics_features(), self.dim + self.hdim)

        # make two-input MLP (one input is existing patch context, the other is transcriptomics, then output is the same size as the patch context)
        # self.combine_transcriptomics_patch_ctx = nn.Sequential(
        #     nn.Linear(self.dim + self.hdim + get_num_transcriptomics_features(), self.dim + self.hdim),
        #     nn.ReLU(),
        #     nn.Linear(self.dim + self.hdim, self.dim + self.hdim)
        # )
        if config.add_transcriptomics:
            if config.transcriptomics_combine_method == "residual_enrichment":
                self.combine_transcriptomics_patch_ctx = CombineTranscriptomicsPatchCtx(
                    feat_1_dim=self.dim, 
                    feat_2_dim=get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path), 
                    hdim=self.hdim,
                    dropout_p=0.2
                )
            elif config.transcriptomics_combine_method == "attention_weighted_sum":
                print(f"feature 1 dim: {self.dim}, feature 2 dim: {get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path)}, hdim: {self.hdim}")
                self.combine_transcriptomics_patch_ctx = AttentionWeightedSum(
                    feat_1_dim=self.dim, 
                    feat_2_dim=get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path), 
                    hidden_dim=self.hdim, 
                )
            elif config.transcriptomics_combine_method == "gated_enrichment":
                # feat1_dim = (self.dim + self.hdim) if config.lstm else self.dim
                # feat2_dim = get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path)
                self.combine_transcriptomics_patch_ctx = GatedTranscriptomicsFusion(
                    feat1_dim=self.dim,
                    feat2_dim=get_num_transcriptomics_features(transcriptomics_model_path=config.transcriptomics_model_path),
                    hidden_dim=self.hdim,
                    dropout_p=0.2
                )
            else:
                raise ValueError(
                    f"Unknown transcriptomics combine method: {config.transcriptomics_combine_method}"
                )

    def process(self, data: PatchBatch, lstm=None) -> Dict:
        """
        Process a batch of slides to produce a batch of logits. The class PatchBatch is defined to manage the complexity
        of each slide having different length etc. (padding is required).
        """
        patch_features = data.fts
        transcriptomics = data.transcriptomics
        patch_features = self.proj_in(patch_features)

        ################# Apply LSTM
        if self.config.lstm:
            assert lstm is not None

            # Initialise LSTM state at top of hierarchy
            if self.depth == 0:
                hs = torch.zeros(
                    (data.batch_size, data.max_patches, self.dim), device=data.device
                )
                cs = torch.zeros(
                    (data.batch_size, data.max_patches, self.hdim), device=data.device
                )

            # Otherwise, retrieve it
            else:
                lstm_state = data.ctx_patch[:, :, -1]
                assert (
                    lstm_state.shape[-1] == self.dim + self.hdim
                    or lstm_state.shape[-1]
                    == self.dim + self.hdim + transcriptomics.shape[-1]
                )
                hs, cs = lstm_state[..., : self.dim], lstm_state[..., self.dim :]

            hs, cs = lstm(patch_features, hs, cs)

            patch_features = patch_features + hs  # produce Y from X

            patch_ctx = torch.concat((hs, cs), dim=-1)
            # print('patch_ctx shape after concatenating hs and cs is ', patch_ctx.shape)
        ################# Get importance values \alpha
        # (this method ensures padding is assigned 0 importance: apply the MLP+sigmoid only to non-background patches)

        # TODO: put this back in later
        # if self.config.add_transcriptomics:
        #     importance = utils.apply_to_non_padded(
        #         lambda xs: torch.sigmoid(self.importance_mlp(torch.concat((xs["contextualised_features"], xs["transcriptomics"].clone().detach()), dim=-1))),
        #         patch_features,
        #         transcriptomics,
        #         data.valid_inds,
        #         1,
        #     )[..., 0]
        # else:
        importance = utils.apply_to_non_padded(
            lambda xs: torch.sigmoid(
                self.importance_mlp(xs["contextualised_features"])
            ),
            patch_features,
            # transcriptomics,
            data.valid_inds,
            1,
        )[..., 0]
        if self.config.importance_mode == "mul":
            # produce Z from Y
            patch_features = patch_features * importance[..., None]

        # If not using the LSTM, apply a RNN instead
        if not self.config.lstm:
            if self.depth > 0 and self.config.hierarchical_ctx:
                assert len(data.ctx_patch.shape) == 4
                hctx = data.ctx_patch[:, :, -1]  # B x MaxIms x D
                hctx = utils.apply_to_non_padded(
                    self.hctx_mlp, hctx, data.valid_inds, self.dim
                )

                patch_features = patch_features + hctx

            patch_ctx = patch_features

        # append transcriptomics features to patch context
        if self.config.add_transcriptomics and (transcriptomics is not None):
            # generate a random tensor of the same shape as the transcriptomics tensor
            # random_tensor = torch.rand(transcriptomics.shape).to(transcriptomics.device)
            # patch_ctx = self.combine_transcriptomics_patch_ctx(
            #     torch.cat((patch_ctx, random_tensor), dim=-1)
            # )

            # print("adding transcriptomics features to patch context")
            # TODO: check if this is correct
            if self.config.transcriptomics_combine_method == "residual_enrichment":
                valid_mask = transcriptomics.abs().sum(dim=-1, keepdim=True) != 0
                # print("mask ", valid_mask)
                enriched_ctx = self.combine_transcriptomics_patch_ctx(
                    feat_1=patch_ctx,
                    feat_2=transcriptomics,
                )
                # enriched_ctx = self.combine_transcriptomics_patch_ctx(
                #     torch.cat((patch_ctx, transcriptomics.clone().detach()), dim=-1)
                # )
                patch_ctx = torch.where(valid_mask, enriched_ctx, patch_ctx)
                # print how many patches are not zero
                print(
                    f"Number of patches with non-zero transcriptomics features: {valid_mask.sum()} out of {valid_mask.numel()}"
                )
            elif self.config.transcriptomics_combine_method == "gated_enrichment":
                patch_ctx = self.combine_transcriptomics_patch_ctx(
                    feat_1=patch_ctx,        # [B, N, D]
                    feat_2=transcriptomics   # [B, N, F]
                )

        ################# Global aggregation
        d = self.config.trans_dim

        # Unused conditional sequence for aggregation. We tried putting slide context here but it performed poorly
        #  compared to residual context.
        encoder_input = torch.zeros((data.batch_size, 0, d), device=data.device)

        # Positional encoding
        xs = patch_features
        patch_locs = data.locs // self.config.patch_size  # pixel coords -> patch coords
        if self.config.pos_encoding_mode == "1d":
            xs = self.global_agg.pos_encode_1d(xs)
        elif self.config.pos_encoding_mode == "2d":
            xs = self.global_agg.pos_encode_2d(xs, patch_locs)

        # Aggregate
        slide_features = self.global_agg(encoder_input, xs, None, data.num_ims)

        # Apply residual connection
        if self.config.slide_ctx_mode == "residual" and data.ctx_depth > 0:
            slide_features = slide_features + data.ctx_slide[:, -1]

        ################# Produce final logits
        if self.config.slide_ctx_mode == "concat":
            all_ctx = torch.flatten(
                data.ctx_slide, start_dim=1
            )  # B x K x D -> B x (K * D)
            ft = torch.cat((all_ctx, slide_features), dim=1)
            logits = self.classification_layer(ft)
        else:
            logits = self.classification_layer(slide_features)

        return {
            "logits": logits,
            "ctx_slide": slide_features,
            "ctx_patch": patch_ctx,  # (actually RNN state)
            "importance": importance,
        }

    def ctx_dim(self) -> Tuple[int, int]:
        if self.config.lstm:
            return self.slide_ctx_dim, self.dim + self.hdim
        return self.slide_ctx_dim, self.dim


def load(slide_id, power: float):
    assert root_dir is not None, f"set_preprocess_dir must be called before load!"
    path = join(root_dir, slide_id + f"_{power:.3f}.pt")
    assert os.path.isfile(path), f"Pre-process load: path '{path}' not found!"
    return torch.load(path)


if __name__ == "__main__":
    print("this runs")

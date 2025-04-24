import json
import os
from os.path import join
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from matplotlib.patches import Rectangle
import xml.etree.ElementTree as ET

from model.transcriptomics_engine import get_transcriptomics_data
import utils
from config import Config
from utils import device
from data_utils.patch_batch import from_raw_slide
from data_utils.slide import load_raw_slide
from model.interface import RecursiveModel
from model.image_encoder import from_name
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import yaml


def load_gene_list(st_model_config_file: str):
    st_model_config = yaml.safe_load(open(st_model_config_file, "r"))
    gene_sample_selection = st_model_config["dataloader_config"][
        "sample_gene_selection"
    ]

    st_config = json.load(open(gene_sample_selection, "r"))
    gene_list = st_config["highly_variable_genes"]
    return gene_list


def parse_camelyon17_anno_file(path: str):
    assert os.path.isfile(path), f"Couldn't find annotation file at '{path}'."
    # Parse the XML file
    tree = ET.parse(path)
    root = tree.getroot()

    # Check the group name
    group_name = root.find(".//Group").get("Name")
    if group_name != "Tumor":
        raise ValueError(f"Unexpected group name: {group_name}")

    # Initialize list to hold polygon data
    polygons = []

    # Iterate over annotations and extract coordinates
    for annotation in root.findall(".//Annotation"):
        if annotation.get("Type") != "Polygon":
            raise ValueError(f"Unexpected annotation type: {annotation.get('Type')}")

        # Get color and coordinates
        color = annotation.get("Color")
        coords = [
            (float(coord.get("X")), float(coord.get("Y")))
            for coord in annotation.find("Coordinates")
        ]

        # Append polygon data to list
        polygons.append((coords, color))

    return polygons

# `pix` pixels at depth `depth` to a different depth
def convert_pix(pix, depth, to_depth):
    e = to_depth - depth
    if e <= 0:
        return pix // 2 ** (-e)
    else:
        return pix * 2**e

def to_pix_space(depth, y, x):
    return convert_pix(y, depth, 0), convert_pix(x, depth, 0)

def plot_full_gene_expression_overlay(
    config: Config,
    slide_depths: list,
    gene_list: list,
    base_img: np.ndarray,
    transform,
    transcriptomics_model_path: str,
    out_path: str,
    image_encoder=None,
    pad: int = 128,
    P: int = 256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    depth = config.num_levels - 1
    # highest_mag_slide = slide_depths[-1]
    
    # start with everything?
    selected_slide = slide_depths[0]
    
    # calculate total WSI dimensions at highest mag
    base_height, base_width = base_img.shape[:2]
    total_width = convert_pix(base_width, 0, depth)
    total_height = convert_pix(base_height, 0, depth)
    
    # generate all possible patch locations at highest mag
    patch_size = config.model_config.patch_size
    x_coords = range(0, total_width, patch_size)
    y_coords = range(0, total_height, patch_size)
    
    # create grid of all possible patches
    all_patch_locs = [(y, x) for y in y_coords for x in x_coords]
    
    # predict gene expression for all patches
    all_patch_tensors = []
    transcriptomics_results = []
    
    invalid_indices = []
    with torch.no_grad():
        for i in range(len(all_patch_locs)):
            try:
                # get patch data directly from raw slide
                patch_data = selected_slide.get_patch_data(all_patch_locs[i])
                patch_tensor = transform(patch_data).unsqueeze(0).to(device)
                all_patch_tensors.append(patch_tensor)
                print("Found patch")
            except Exception as e:
                # print(f"Error processing patch {patch_loc}: {str(e)}")
                invalid_indices.append(i)
                continue

    # remove invalid indices from all_patch_locs and all_patch_tensors
    all_patch_locs = [loc for i, loc in enumerate(all_patch_locs) if i not in invalid_indices]
    print(f"Found {len(all_patch_locs)} valid patches")

    print(len(selected_slide.locs))

    # patches is a list of tensors, each with shape (1, C, P, P)
    # Concatenate them into a single tensor
    # Shape: (N, C, P, P)
    patches = torch.cat(all_patch_tensors)
    print(f"Shape of patches before encoder: {patches.shape}")

    # run the encoder
    patches = image_encoder(patches)
    print(f"Shape of patches after encoder: {patches.shape}")

    transcriptomics_results = get_transcriptomics_data(
        patches, transcriptomics_model_path=transcriptomics_model_path
    )
    
    for gene_idx, gene_name in enumerate(gene_list):
        print(f"Plotting {gene_name} expression")
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(base_img, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # normalize expressions using robust scaling
        expressions = transcriptomics_results[:, gene_idx].cpu()
        vmin, vmax = np.quantile(expressions, [0.05, 0.95])
        norm = plt.Normalize(vmin, vmax)
        cmap = plt.get_cmap("viridis")
        
        # plot all patches
        for (patch_loc, expr) in zip(all_patch_locs, expressions):
            y_base, x_base = to_pix_space(depth, *patch_loc)
            patch_size_px = convert_pix(P, depth, 0)
            
            rect = Rectangle(
                (x_base, y_base), patch_size_px, patch_size_px,
                facecolor=cmap(norm(expr)),
                edgecolor="none",
                alpha=0.6
            )
            ax.add_patch(rect)
            
        # add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        fig.colorbar(sm, cax=cax, label=f"{gene_name} Expression Level")
        
        # save output
        if out_path:
            safe_name = "".join([c if c.isalnum() else "_" for c in gene_name])
            plt.savefig(
                f"{out_path}_FULL_{safe_name}.pdf"
            )
        
        plt.close(fig)

def plot_gene_expression_overlay(
    config: Config,
    slide_depths: list,
    gene_list: list,
    base_img: np.ndarray,
    transform,
    transcriptomics_model_path: str,
    out_path: str = None,
    image_encoder=None,
    pad: int = 128,
    P: int = 256,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    highest_mag_slide = slide_depths[-1]
    highest_mag_patches = highest_mag_slide.locs
    depth = config.num_levels - 1

    transcriptomics_results = []
    patches = []
    with torch.no_grad():
        for patch_loc in highest_mag_patches:
            print(patch_loc)
            patch_data = highest_mag_slide.get_patch_data(patch_loc)
            patches.append(transform(patch_data))
            print(f"Shape of added patch: {patches[-1].shape}")

    patches = [patch.unsqueeze(0).to(device) for patch in patches]
    # patches is a list of tensors, each with shape (1, C, P, P)
    # Concatenate them into a single tensor
    # Shape: (N, C, P, P)
    patches = torch.cat(patches)

    # run the encoder
    patches = image_encoder(patches)
    print(f"Shape of patches after encoder: {patches.shape}")

    transcriptomics_results = get_transcriptomics_data(
        patches, transcriptomics_model_path=transcriptomics_model_path
    )

    # visualization loop
    for gene_idx, gene_name in enumerate(gene_list):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(base_img)
        ax.axis("off")

        expressions = transcriptomics_results[:, gene_idx].cpu()
        vmin, vmax = np.quantile(expressions, [0.05, 0.95])
        norm = plt.Normalize(vmin, vmax)
        cmap = plt.get_cmap("inferno")

        for patch_loc, expr in zip(highest_mag_patches, expressions):
            y_base, x_base = to_pix_space(depth, *patch_loc)
            patch_size = convert_pix(P, depth, 0)

            rect = Rectangle(
                (x_base, y_base),
                patch_size,
                patch_size,
                facecolor=cmap(norm(expr)),
                edgecolor="red",
                linewidth=0.8,
                alpha=0.7,
            )
            ax.add_patch(rect)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label(f"{gene_name} Expression", rotation=270, labelpad=15)

        y_coords = [to_pix_space(depth, loc[0], 0)[0] for loc in highest_mag_patches]

        if y_coords:
            ax.set_ylim(max(y_coords) + pad + P, min(y_coords) - pad)

        if out_path is not None:
            safe_gene_name = "".join(c if c.isalnum() else "_" for c in gene_name)

            gene_path = f"{out_path}_{safe_gene_name}_expression.pdf"
            plt.savefig(gene_path, format="pdf", dpi=300, bbox_inches="tight")
            print(f"Saved {gene_name} expression plot to {gene_path}")

        plt.close(fig)


def heatmap_camelyon17_transcriptomics(
    config: Config,
    model: RecursiveModel,
    image_encoder,
    transform,
    slide_path: str,
    annotation_path: str,
    out_path: str,
    gene_list: list,
    transcriptomics_model_path: str,
):
    # check if the WSI exists
    assert os.path.isfile(slide_path), f"Couldn't find WSI at path '{slide_path}'."

    # create the output directory if necessary
    if out_path is not None:
        directory = os.path.join(*os.path.split(out_path)[:-1])
        if directory != "" and not os.path.isdir(directory):
            print("Creating directory:", directory)
            os.makedirs(directory, exist_ok=True)

    if annotation_path is not None:
        assert os.path.isfile(annotation_path), (
            f"Couldn't find annotation XML file at path '{annotation_path}'. The "
            f"annotation file is optional, but if the argument is passed, "
            f"the file must exist."
        )

    def get_slide_rgb():
        return slide.view_at_power(config.base_power)

    L = config.num_levels
    P = config.model_config.patch_size

    slide = load_raw_slide(
        slide_path,
        config.base_power,
        config.model_config.patch_size,
        model.procs[0].ctx_dim(),
        prepatch=False,
        tissue_threshold=0.025,
    )

    slide.camelyon = True
    slide.load_patches()

    slide_depths = [slide]
    imps = []

    print("Recursing...")
    for depth in range(config.num_levels):
        print(f" Depth {depth + 1} / {config.num_levels}...")
        data = from_raw_slide(
            slide,
            image_encoder,
            transform,
            transcriptomics_model_path=transcriptomics_model_path,
        )
        out = model(depth, data)
        ctx_slide = out["ctx_slide"][0]
        ctx_patch = out["ctx_patch"][0]
        importance = out["importance"][0]
        imps.append(importance.detach().cpu().numpy())

        if depth != config.num_levels - 1:
            slide = slide.recurse(
                config.magnification_factor,
                ctx_slide.cpu(),
                ctx_patch.cpu(),
                importance.cpu(),
                config.top_k_patches[depth],
            )
            slide.camelyon = True
            slide.load_patches()
            slide_depths.append(slide)

    # print the number of locs at each depth
    for depth in range(config.num_levels):
        print(f"Depth {depth + 1} / {config.num_levels}: {slide_depths[depth].locs.shape[0]}")

    os._exit(0)

    # get the base image for visualization
    bigimg = get_slide_rgb()

    plot_full_gene_expression_overlay(
        config=config,
        slide_depths=slide_depths,
        gene_list=gene_list,
        base_img=bigimg,
        transform=transform,
        transcriptomics_model_path=transcriptomics_model_path,
        out_path=out_path,
        image_encoder=image_encoder,
        pad=128,
        P=config.model_config.patch_size,
    )

    H, W, C = bigimg.shape
    assert C == 3
    print("Bigimg:", H, "x", W, "x", C)

    ns = [s.locs.shape[0] for s in slide_depths]
    print("Patch counts:", ns)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3.4))

    sax = axes[0]
    sax.imshow(bigimg, aspect="equal")
    sax.set_xticks([])
    sax.set_yticks([])

    if annotation_path is not None:
        print("Plotting CAMELYON17 slide annotations")
        try:
            polygons = parse_camelyon17_anno_file(annotation_path)
        except Exception as e:
            raise ValueError(
                f"Failed to parse CAMELYON17 annotation path at '{annotation_path}'."
            ) from e

        multiplier = config.base_power / 40
        for coords, color in polygons:
            x, y = zip(*coords)
            x = [i * multiplier for i in x]
            y = [i * multiplier for i in y]
            axes[0].plot(
                x + [x[0]], y + [y[0]], color="blue", linewidth=2
            )  # Close the loop

    ax = axes[1]
    ax.imshow(bigimg, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])

    shape = (H, W)
    overall_imp = np.zeros((L,) + shape)  # 0 = padding

    s1, s2 = shape

    # DRAW WIREFRAME RECTS
    for depth in range(L):
        locs = slide_depths[depth].locs
        size = convert_pix(P, depth, 0)
        lw = 0.5

        imp = imps[depth]

        for i in range(locs.shape[0]):
            y, x = locs[i].tolist()
            y, x = to_pix_space(depth, y, x)
            rect = Rectangle(
                (x, y), size, size, facecolor="None", edgecolor="black", lw=lw
            )
            ax.add_patch(rect)

            overlap_y = max(y, 0) <= min(y + size, s1)
            overlap_x = max(x, 0) <= min(x + size, s2)
            if overlap_y and overlap_x:
                y1 = max(y, 0)
                y2 = min(y + size, s1)
                x1 = max(x, 0)
                x2 = min(x + size, s2)
                overall_imp[depth, y1:y2, x1:x2] = imp[i] + 1e-4

    # Weight importances by 1/2^depth
    for depth in range(L - 2, -1, -1):
        relevant_mask = overall_imp[depth + 1] != 0
        overall_imp[depth][relevant_mask] = (
            overall_imp[depth][relevant_mask]
            + overall_imp[depth + 1][relevant_mask] * 0.5
        )

    overall_imp = overall_imp[0]

    alpha = np.where(overall_imp > 0, 0.5, 0)
    overall_imp[overall_imp == 0] = np.min(overall_imp[overall_imp > 0])

    print(bigimg.shape, overall_imp.shape)
    hm = ax.imshow(overall_imp, cmap="viridis", alpha=alpha, aspect="equal")

    # Try to choose an appropriate viewport: look at the positions of patches
    y = slide_depths[0].locs[:, 0].tolist()
    h = bigimg.shape[0]

    # exclude top/bottom patches: if present, these are often background removal failures
    thresh = 0.1
    y = [i for i in y if 0.1 < (i + P / 2) / h < 1 - thresh]

    pad = 128
    axes[0].set_ylim(max(y) + pad + P, min(y) - pad)
    axes[1].set_ylim(max(y) + pad + P, min(y) - pad)

    cax = inset_axes(axes[1], width="5%", height="100%", loc="right", borderpad=-1.5)
    fig.colorbar(hm, cax=cax, orientation="vertical")

    fig.tight_layout()
    fig.subplots_adjust(right=0.9)

    if out_path is not None:
        if not out_path.endswith(".pdf"):
            out_path += ".pdf"
        plt.savefig(out_path, format="pdf", dpi=200)
    plt.show()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model-dir",
        required=True,
        type=str,
        help="Path to model directory. Must contain config.json file.",
    )
    parser.add_argument(
        "-s", "--slide-path", required=True, type=str, help="Path to the WSI."
    )
    parser.add_argument(
        "-a",
        "--annotation-path",
        default=None,
        type=str,
        help="For CAMELYON17 slides, the path to the XML annotation file. Do not pass this argument for non-CAMELYON17 slides, or slides without an annotation file.",
    )
    parser.add_argument(
        "-c",
        "--st-model-config",
        default=None,
        type=str,
        help="Path to the transcriptomics model config file. This is used to load the gene list.",
    )
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        type=str,
        help="Output a PDF of the visualisation to the given path.",
    )
    args = parser.parse_args()

    model_name = os.path.split(args.model_dir)[-1]

    config = Config.load(
        args.model_dir, test_mode=True
    )  # test_mode stops error when checking existence of data dirs

    print(config)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    model = config.get_model()
    model = model.eval().to(device)
    train_stats = utils.load_state(args.model_dir, model, map_location=device)
    print("Loaded from epoch", train_stats["epoch"])

    name = os.path.split(args.model_dir)[-1]

    image_encoder, dimension, transform = from_name("UNI")
    image_encoder.to(device)

    heatmap_camelyon17_transcriptomics(
        config,
        model,
        image_encoder,
        transform,
        args.slide_path,
        args.annotation_path,
        args.out,
        gene_list=load_gene_list(args.st_model_config),
        transcriptomics_model_path="",
    )

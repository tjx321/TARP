"""TARP: Text-Anchored Re-centering Projection for few-shot CLIP adaptation.

Unified entry point for all CLIP backbones. The TARP pipeline is:

    text prototype (averaged text descriptions)
        + extra text-as-shot cache (count=50, aggregate=mean)
        + re-centered FLD projection (fld_center, text_prior_tau=4)
        + validation-split search of the cache-balance coefficient.

Usage:
    python main.py --config configs/eurosat/16shot.yaml --backbone RN50 --seed 1

The cache directory is derived from ``--backbone`` and ``--seed``:
    caches/<backbone_lower>/seed{seed}/<dataset>/

Set ``load_cache: true`` and ``load_pre_feat: true`` in the config to reuse the
precomputed caches (see scripts/download_caches.sh); otherwise they are
recomputed and stored under the same directory.
"""

import argparse
import importlib.util
import json
import os
import random
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml

import clip
from datasets import build_dataset
from datasets.utils import build_data_loader

# Import the re-centered FLD projector (modules/fld_base.py).
_fld_module_path = os.path.join(os.path.dirname(__file__), "modules", "fld_base.py")
_spec = importlib.util.spec_from_file_location("fld_base", _fld_module_path)
_fld_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fld_module)
fld_center = _fld_module.fld_center

from utils import (
    cls_acc,
    build_clip_cache_model,
    pre_CLIP_load_features,
    compute_class_means_with_text_prior,
    build_text_prototype_weights,
    build_text_shot_cache,
)

# ---- Fixed method hyper-parameters ----
TEXT_PRIOR_TAU = 4.0           # tau: text-prior strength, dataset-independent

# Supported CLIP backbones -> cache directory tag.
BACKBONES = {
    "RN50": "rn50",
    "RN101": "rn101",
    "ViT-B/16": "vitb16",
    "ViT-B/32": "vitb32",
}


def get_arguments():
    parser = argparse.ArgumentParser(description="TARP few-shot CLIP adaptation.")
    parser.add_argument("--config", required=True, help="path to the dataset/shot yaml config")
    parser.add_argument("--backbone", default="RN50", choices=list(BACKBONES.keys()),
                        help="CLIP backbone")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--exp_name", default="tarp_exp", help="experiment name for the result folder")
    parser.add_argument("--cache_root", default="./caches", help="root directory of precomputed caches")
    return parser.parse_args()


def run_ensemble(cfg, clip_cache_keys, clip_cache_values, val_clip_features,
                 test_clip_features, val_labels, test_labels, clip_weights):
    save_dir = cfg.get("save_dir")
    if save_dir is None:
        raise RuntimeError("cfg['save_dir'] not set.")

    logpath = os.path.join(save_dir, "log.txt")

    def log(msg):
        print(msg)
        with open(logpath, "a") as lf:
            lf.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    log("Starting run_ensemble: text_prototype + text_shot(50, mean) + fld_center (TARP).")

    num_classes = clip_weights.shape[1]
    clip_labels = clip_cache_values.argmax(dim=1)
    clip_cache_keys_T = clip_cache_keys.T.float()

    # ---- Re-centered FLD projection ----
    log(f"Computing fld_center projection with text_prior_tau={TEXT_PRIOR_TAU}.")
    W_clip = fld_center(
        clip_cache_keys_T,
        clip_labels,
        num_classes,
        clip_weights.T,
        text_prior_tau=TEXT_PRIOR_TAU,
    )
    log(f"W_clip shape: {tuple(W_clip.shape)}")

    cache_dir = cfg["cache_dir"]
    W_save_name = f"seed{cfg['seed']}_fld_center_ex_{cfg['shots']}shots.pt"
    torch.save(W_clip, os.path.join(cache_dir, W_save_name))
    log(f"Saved W_clip to {os.path.join(cache_dir, W_save_name)}")

    # ---- Re-centered class means in the FLD subspace ----
    _, _, _, clip_class_means, _ = compute_class_means_with_text_prior(
        clip_cache_keys_T.float(),
        clip_labels,
        num_classes,
        clip_weights.T,
        W_clip=W_clip,
        text_prior_tau=TEXT_PRIOR_TAU,
    )
    clip_class_means = F.normalize(clip_class_means, dim=1)

    # ---- Validation-split hyperparameter search ----
    log("-------- Computing validation set features for hyperparameter search. --------")
    val_clip_fld = val_clip_features.float() @ W_clip
    val_clip_dist = torch.cdist(val_clip_fld, clip_class_means)
    val_clip_fld_logits = 1 / (val_clip_dist + 1e-6)
    val_clip_logits = 100.0 * val_clip_features.float() @ clip_weights.float()

    log("-------- Searching hyperparameters on validation set. --------")
    # logits = clip_logit + scale * (alpha2 / alpha1) * fld_logit
    best_val_acc = -1.0
    best_scale = 1.0
    best_alpha1 = best_alpha2 = 1

    scale_list = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    alpha_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    total_combinations = len(scale_list) * len(alpha_list) ** 2
    current_combination = 0

    for scale in scale_list:
        for alpha1 in alpha_list:
            for alpha2 in alpha_list:
                current_combination += 1
                val_logits = val_clip_logits + scale * (alpha2 / alpha1) * val_clip_fld_logits
                acc = cls_acc(val_logits, val_labels)
                if acc >= best_val_acc:
                    best_val_acc = acc
                    best_scale, best_alpha1, best_alpha2 = scale, alpha1, alpha2
                    log(f"[{current_combination}/{total_combinations}] New best: "
                        f"scale={scale:.1f}, alpha1={alpha1}, alpha2={alpha2}, val_acc={acc:.2f}")

    log(f"Best hyperparameters: scale={best_scale:.1f}, alpha1={best_alpha1}, alpha2={best_alpha2}")
    log(f"Best validation accuracy: {best_val_acc:.2f}")

    # ---- Test-set evaluation ----
    log("-------- Evaluating on the test set with best hyperparameters. --------")
    test_clip_fld = test_clip_features.float() @ W_clip
    clip_dist_test = torch.cdist(test_clip_fld, clip_class_means)
    clip_fld_logits_test = 1 / (clip_dist_test + 1e-6)
    test_clip_logits = 100.0 * test_clip_features.float() @ clip_weights.float()

    final_logits = test_clip_logits + best_scale * (best_alpha2 / best_alpha1) * clip_fld_logits_test
    final_acc = cls_acc(final_logits.cpu(),
                        test_labels.cpu() if test_labels.is_cuda else test_labels)
    cache_acc = cls_acc(clip_fld_logits_test.cpu(),
                        test_labels.cpu() if test_labels.is_cuda else test_labels)
    log(f"**** Final test accuracy: {final_acc:.2f}. ****")
    log(f"**** Cache-only fld_center accuracy: {cache_acc:.2f}. ****")

    result = {
        "dataset": cfg.get("dataset"),
        "shots": cfg.get("shots"),
        "backbone": cfg.get("clip_backbone"),
        "text_source": "text_prototype",
        "fld_projector": "fld_center",
        "text_prior_tau": TEXT_PRIOR_TAU,
        "extra_text_shots": cfg["extra_text_shots"],
        "best_hyperparameters": {"scale": float(best_scale),
                                 "alpha1": int(best_alpha1), "alpha2": int(best_alpha2)},
        "val_acc": float(best_val_acc),
        "cache_acc": float(cache_acc),
        "test_acc": float(final_acc),
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
    }
    with open(os.path.join(save_dir, "results.json"), "w") as rf:
        json.dump(result, rf, indent=2)
    log(f"Saved results.json to {save_dir}")
    log("Run finished.\n")


def main():
    args = get_arguments()
    if not os.path.exists(args.config):
        print(f"Error: config not found: {args.config}")
        sys.exit(1)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    # Resolve ${VAR} environment placeholders (e.g. ${DATASET_ROOT}) in paths.
    cfg["root_path"] = os.path.expandvars(cfg["root_path"])
    cfg["seed"] = args.seed
    cfg["clip_backbone"] = args.backbone

    backbone_tag = BACKBONES[args.backbone]
    # Unified cache layout: caches/<backbone>/seed{S}/<dataset>/
    cache_dir = os.path.join(args.cache_root, backbone_tag, f"seed{args.seed}", cfg["dataset"])
    os.makedirs(cache_dir, exist_ok=True)
    cfg["cache_dir"] = cache_dir

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_root = os.path.join("./results", backbone_tag)
    base_save_dir = os.path.join(save_root, cfg["dataset"], f"{cfg['shots']}shot",
                                 f"{args.exp_name}_{timestamp}")
    os.makedirs(base_save_dir, exist_ok=True)
    cfg["save_dir"] = base_save_dir

    with open(os.path.join(base_save_dir, "config_used.yaml"), "w") as cf:
        yaml.dump(cfg, cf)

    logpath = os.path.join(base_save_dir, "main.log")

    def log_main(msg):
        print(msg)
        with open(logpath, "a") as lf:
            lf.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")

    log_main(f"\nRunning TARP (fld_center): backbone={args.backbone}, "
             f"text_prior_tau={TEXT_PRIOR_TAU}, load_cache={cfg.get('load_cache')}, "
             f"load_pre_feat={cfg.get('load_pre_feat')}")
    log_main(str(cfg) + "\n")

    clip_model, preprocess = clip.load(cfg["clip_backbone"])
    clip_model.eval()

    print(f"Setting seed to {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_main("Preparing dataset.")
    dataset = build_dataset(cfg["dataset"], cfg["root_path"], cfg["shots"])

    val_loader = build_data_loader(data_source=dataset.val, batch_size=64,
                                   is_train=False, tfm=preprocess, shuffle=False)
    test_loader = build_data_loader(data_source=dataset.test, batch_size=64,
                                    is_train=False, tfm=preprocess, shuffle=False)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711)),
    ])
    train_loader_cache = build_data_loader(data_source=dataset.train_x, batch_size=256,
                                           tfm=train_transform, is_train=True, shuffle=False)

    # ---- Text shots: per-class text descriptions appended to the support cache ----
    cfg["extra_text_shots"] = {"count": 50, "include_base_prompt": False}

    # ---- Text prototype: averaged multi-description prompts ----
    log_main("Building text prototypes from averaged multi-description prompts.")
    clip_weights = build_text_prototype_weights(dataset.classnames, clip_model,
                                                cfg["dataset"], cfg)

    # ---- CLIP image cache (few-shot support set) ----
    log_main("Constructing CLIP cache model.")
    clip_cache_keys, clip_cache_values = build_clip_cache_model(cfg, clip_model, train_loader_cache)

    # ---- Extra text-as-shot cache (count=50, mean) ----
    text_cache_keys, text_cache_values = build_text_shot_cache(dataset.classnames, clip_model,
                                                               cfg["dataset"], cfg)
    log_main(f"Text-shot cache shape: keys={tuple(text_cache_keys.shape)}, "
             f"values={tuple(text_cache_values.shape)}")

    clip_cache_keys = clip_cache_keys.float()
    clip_cache_values = clip_cache_values.float()

    # ---- Load val/test features ----
    log_main("Loading CLIP features from val set.")
    val_clip_features, val_labels = pre_CLIP_load_features(cfg, "val", clip_model, val_loader)
    log_main("Loading CLIP features from test set.")
    test_clip_features, test_labels = pre_CLIP_load_features(cfg, "test", clip_model, test_loader)

    # ---- Run ensemble ----
    # Concatenate the text-shot cache onto the few-shot image cache.
    text_cache_keys = text_cache_keys.to(device=clip_cache_keys.device, dtype=clip_cache_keys.dtype)
    text_cache_values = text_cache_values.to(device=clip_cache_values.device, dtype=clip_cache_values.dtype)

    augmented_clip_cache_keys = torch.cat([clip_cache_keys, text_cache_keys], dim=1)
    augmented_clip_cache_values = torch.cat([clip_cache_values, text_cache_values], dim=0)
    log_main(f"Augmented CLIP cache shape: keys={tuple(augmented_clip_cache_keys.shape)}, "
             f"values={tuple(augmented_clip_cache_values.shape)}")

    run_ensemble(cfg, augmented_clip_cache_keys, augmented_clip_cache_values,
                 val_clip_features, test_clip_features, val_labels, test_labels,
                 clip_weights)


if __name__ == "__main__":
    main()

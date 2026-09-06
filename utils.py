"""TARP utilities: cache IO, class means with text prior, text prototypes, accuracy."""

import os
import re
import json
import importlib.util

from tqdm import tqdm

import torch
import torch.nn.functional as F

import clip

# Dataset-name mappings for the text corpus (EuroSAT class map, fgvc corpus key).
_name_map_path = os.path.join(
    os.path.dirname(__file__), "datasets", "text_descriptions", "name_map.py"
)
_spec = importlib.util.spec_from_file_location("name_map", _name_map_path)
_name_map = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_name_map)
EUROSAT_CLASSNAME_MAP = _name_map.EUROSAT_CLASSNAME_MAP
DATASET_TO_CORPUS = _name_map.DATASET_TO_CORPUS


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def compute_class_means_with_text_prior(
    feat,
    label,
    num_classes,
    text_prototypes,
    W_clip=None,
    text_prior_tau=4.0,
    normalize_proto=False,
):
    """Class means of image features fused with the CLIP text prototypes.

    mu_c = (n_c / (n_c + tau)) * mu_img_c + (tau / (n_c + tau)) * mu_txt_c

    If W_clip is given, features and prototypes are first projected into the FLD
    subspace (pass clip_weights.T when only a [D, C] matrix is available).
    Returns (class_means_img, class_sizes, N, class_means, txt_class_means).
    """
    device = feat.device
    dtype = feat.dtype
    D = feat.shape[1]

    if W_clip is not None:
        W_clip = W_clip.to(device=device, dtype=torch.float32)
        feat_work = feat.to(device=device, dtype=torch.float32) @ W_clip  # [N, K]
        dim_work = W_clip.shape[1]
    else:
        feat_work = feat
        dim_work = D

    class_means_img = []
    class_sizes_list = []
    for c in range(num_classes):
        mask = label == c
        n_c = mask.sum().item()
        class_sizes_list.append(n_c)
        if n_c > 0:
            class_means_img.append(feat_work[mask].mean(dim=0))
        else:
            class_means_img.append(torch.zeros(dim_work, device=device, dtype=dtype))
    class_means_img = torch.stack(class_means_img, dim=0)  # [C, dim_work]
    class_sizes = torch.tensor(class_sizes_list, device=device, dtype=dtype)  # [C]
    N = class_sizes.sum()

    text_prototypes = text_prototypes.to(device=device, dtype=dtype)
    if W_clip is not None:
        txt_class_means = (text_prototypes.float() @ W_clip.float()).to(device=device, dtype=dtype)
    else:
        txt_class_means = text_prototypes

    tau = torch.tensor(float(text_prior_tau), device=device, dtype=dtype)
    img_ratio = class_sizes / (class_sizes + tau + 1e-12)  # [C]
    txt_ratio = tau / (class_sizes + tau + 1e-12)          # [C]
    class_means = (
        img_ratio.unsqueeze(1) * class_means_img
        + txt_ratio.unsqueeze(1) * txt_class_means
    )

    if normalize_proto:
        class_means_img = F.normalize(class_means_img, dim=1)
        class_means = F.normalize(class_means, dim=1)
        txt_class_means = F.normalize(txt_class_means, dim=1)

    return class_means_img, class_sizes, N, class_means, txt_class_means


def _load_text_resources(dataset_name, cfg):
    """Load (dataset_key, prompt_template, descriptions) for the text corpus."""
    descriptor_path = cfg.get(
        "descriptor_path",
        os.path.join(
            os.path.dirname(__file__),
            "datasets",
            "text_descriptions",
            "descriptions",
            "image_datasets",
        ),
    )

    cls_to_names_path = os.path.join(
        os.path.dirname(__file__),
        "datasets",
        "text_descriptions",
        "data",
        "cls_to_names.py",
    )
    spec = importlib.util.spec_from_file_location("cls_to_names", cls_to_names_path)
    cls_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cls_mod)
    custom_templates = cls_mod.CUSTOM_TEMPLATES

    dataset_key = DATASET_TO_CORPUS.get(dataset_name, dataset_name)
    if dataset_key not in custom_templates:
        raise KeyError(
            f"Text template is missing for dataset={dataset_name!r} (resolved key: {dataset_key!r})."
        )

    description_file = os.path.join(descriptor_path, f"{dataset_key}.json")

    if not os.path.exists(description_file):
        raise FileNotFoundError(f"Text description file not found: {description_file}")

    with open(description_file, "r", encoding="utf-8") as f:
        descriptions = json.load(f)

    return dataset_key, custom_templates[dataset_key], descriptions


def _normalize_name(name):
    name = name.replace("_", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    name = name.lower().replace("&", " and ")
    name = re.sub(r"[^a-z0-9]+", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _resolve_descriptions(descriptions, dataset_key, classname):
    """Return (matched_key, descriptions_list) for a dataset classname.

    For 10 of the 11 datasets the classname matches a corpus key exactly; EuroSAT
    needs the explicit map, and the normalized comparison below is the fallback.
    """
    if classname in descriptions:
        return classname, descriptions[classname]
    if dataset_key == "eurosat":
        mapped = EUROSAT_CLASSNAME_MAP.get(classname)
        if mapped in descriptions:
            return mapped, descriptions[mapped]
    norm = _normalize_name(classname)
    for key in descriptions:
        if _normalize_name(key) == norm:
            return key, descriptions[key]
    raise KeyError(
        f"No text descriptions for classname={classname!r} (dataset={dataset_key!r}). "
        f"Available keys sample: {list(descriptions)[:10]}"
    )


def _aggregate_text_features(text_features):
    """Average all text descriptions of a class into a single feature."""
    return F.normalize(text_features.mean(dim=0, keepdim=True), dim=-1)


def _get_extra_text_shot_cfg(cfg):
    extra_cfg = cfg["extra_text_shots"]
    return {
        "count": int(extra_cfg["count"]),
        "include_base_prompt": bool(extra_cfg["include_base_prompt"]),
    }


def _encode_class_prompts(classname, descriptions, template, count, include_base_prompt,
                          clip_model, device):
    """Encode `count` text descriptions (plus optional base prompt) for one class."""
    if len(descriptions) < count:
        raise ValueError(
            f"Text corpus only provides {len(descriptions)} descriptions for class {classname!r}, "
            f"but count={count}."
        )
    prompt = template.format(classname.replace("_", " "))
    prompts = []
    if include_base_prompt:
        prompts.append(prompt + ".")
    for i in range(count):
        prompts.append(prompt + ". " + descriptions[i])
    prompts = [p for p in prompts if p.strip()]
    if not prompts:
        raise ValueError(f"No valid text prompts were built for class {classname!r}")

    text_tokens = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
    text_features = clip_model.encode_text(text_tokens).float()
    return F.normalize(text_features, dim=-1)


def build_text_shot_cache(classnames, clip_model, dataset_name, cfg):
    """Encode per-class text descriptions as extra 'text shots' for the support cache."""
    dataset_key, template, descriptions = _load_text_resources(dataset_name, cfg)
    extra_cfg = _get_extra_text_shot_cfg(cfg)
    count = extra_cfg["count"]
    include_base_prompt = extra_cfg["include_base_prompt"]
    if count < 1:
        raise ValueError("text_shot_count must be >= 1")

    device = next(clip_model.parameters()).device
    cache_keys = []
    cache_labels = []

    with torch.no_grad():
        for class_idx, classname in enumerate(classnames):
            prompt_classname, class_descriptions = _resolve_descriptions(
                descriptions, dataset_key, classname
            )
            text_features = _encode_class_prompts(
                prompt_classname, class_descriptions, template, count,
                include_base_prompt, clip_model, device,
            )
            text_features = _aggregate_text_features(text_features)

            cache_keys.append(text_features)
            cache_labels.append(
                torch.full((text_features.shape[0],), class_idx, device=device, dtype=torch.long)
            )

    cache_keys = torch.cat(cache_keys, dim=0)
    cache_keys = F.normalize(cache_keys, dim=-1).T.contiguous()
    cache_values = F.one_hot(torch.cat(cache_labels, dim=0), num_classes=len(classnames)).float()
    return cache_keys, cache_values


def build_text_prototype_weights(classnames, clip_model, dataset_name, cfg):
    """Build the CLIP zero-shot classifier by averaging text descriptions per class."""
    dataset_key, template, descriptions = _load_text_resources(dataset_name, cfg)
    num_descriptor = int(cfg.get("num_descriptor", 50))
    include_base_prompt = bool(cfg.get("include_base_prompt", True))
    if num_descriptor < 1:
        raise ValueError("num_descriptor must be >= 1")

    device = next(clip_model.parameters()).device
    protos = []

    with torch.no_grad():
        for classname in classnames:
            prompt_classname, class_descriptions = _resolve_descriptions(
                descriptions, dataset_key, classname
            )
            class_embs = _encode_class_prompts(
                prompt_classname, class_descriptions, template, num_descriptor,
                include_base_prompt, clip_model, device,
            )
            class_embedding = F.normalize(class_embs.mean(dim=0, keepdim=True), dim=-1).squeeze(0)
            protos.append(class_embedding)

    return torch.stack(protos, dim=1).to(device)


@torch.no_grad()
def build_clip_cache_model(cfg, clip_model, train_loader_cache):
    seed = cfg.get('seed', 1)
    cache_keys_path = os.path.join(cfg['cache_dir'], f'seed{seed}_clip_keys_{cfg["shots"]}shots.pt')
    cache_values_path = os.path.join(cfg['cache_dir'], f'seed{seed}_clip_values_{cfg["shots"]}shots.pt')

    if cfg['load_cache']:
        cache_keys = torch.load(cache_keys_path)
        cache_values = torch.load(cache_values_path)
        return cache_keys, cache_values

    cache_keys = []
    cache_values = []
    for augment_idx in range(cfg['augment_epoch']):
        train_features = []
        print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
        for images, target in tqdm(train_loader_cache):
            images = images.cuda()
            image_features = clip_model.encode_image(images)
            train_features.append(image_features)
            cache_values.append(target.cuda())
        cache_keys.append(torch.cat(train_features, dim=0))

    cache_keys = torch.cat(cache_keys, dim=0)  # [N * augment_epoch, feature_dim]
    cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
    cache_keys = cache_keys.permute(1, 0)  # [feature_dim, N * augment_epoch]
    cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

    torch.save(cache_keys, cache_keys_path)
    torch.save(cache_values, cache_values_path)
    return cache_keys, cache_values


@torch.no_grad()
def pre_CLIP_load_features(cfg, split, clip_model, loader):
    seed = cfg.get('seed', 1)
    features_path = os.path.join(cfg['cache_dir'], f"seed{seed}_{split}_clip_f.pt")
    labels_path = os.path.join(cfg['cache_dir'], f"seed{seed}_{split}_clip_l.pt")

    if cfg['load_pre_feat']:
        features = torch.load(features_path)
        labels = torch.load(labels_path)
        return features, labels

    features, labels = [], []
    for images, target in tqdm(loader):
        images, target = images.cuda(), target.cuda()
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        features.append(image_features)
        labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)
    torch.save(features, features_path)
    torch.save(labels, labels_path)
    return features, labels

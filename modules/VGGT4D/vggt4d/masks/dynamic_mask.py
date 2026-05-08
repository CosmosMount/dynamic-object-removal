import numpy as np
import torch
from einops import rearrange
from skimage.filters import threshold_multiotsu
from sklearn.cluster import KMeans
from tqdm import tqdm


def extract_mean1_map(ref_id, global_q: torch.Tensor, global_k: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # mean q_ref_q_src 3-8
    window = torch.tensor([-6, -4, -2, 2, 4, 6])
    n_img = global_q.shape[0]
    img_h, img_w = images.shape[-2:]
    n_h, n_w = img_h // 14, img_w // 14

    src_ids = ref_id + window

    src_ids = src_ids[src_ids >= 0]
    src_ids = src_ids[src_ids < n_img]
    # print(src_ids)

    layer_ids = torch.arange(3, 8)

    q_ref = global_q[ref_id]
    k_ref = global_k[ref_id]
    # print(q_ref.shape)
    # print(k_ref.shape)
    q_ref = q_ref.unsqueeze(0)[:, layer_ids]
    k_ref = k_ref.unsqueeze(0)[:, layer_ids]

    q_src = global_q[src_ids]
    k_src = global_k[src_ids]
    # print(q_src.shape)
    # print(k_src.shape)
    q_src = q_src[:, layer_ids]
    k_src = k_src[:, layer_ids]
    # print(q_src.shape)
    # print(k_src.shape)

    attn_map = q_ref @ q_src.transpose(-2, -1)
    # print(attn_map.shape)
    attn_map = rearrange(
        attn_map, "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok", n_h=n_h, n_w=n_w)
    # print(attn_map.shape)
    attn_map = attn_map.mean(dim=(2, 3, 4))
    attn_min = attn_map.min()
    attn_max = attn_map.max()

    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    # print(attn_map.shape)

    return attn_map


def extract_spacial_var1_map(ref_id, global_q: torch.Tensor, global_k: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # spacial std q_ref_q_src 18-20
    window = torch.tensor([-6, -4, -2, 2, 4, 6])
    n_img = global_q.shape[0]
    img_h, img_w = images.shape[-2:]
    n_h, n_w = img_h // 14, img_w // 14

    src_ids = ref_id + window

    src_ids = src_ids[src_ids >= 0]
    src_ids = src_ids[src_ids < n_img]
    # print(src_ids)

    layer_ids = torch.arange(18, 20)

    q_ref = global_q[ref_id]
    k_ref = global_k[ref_id]
    # print(q_ref.shape)
    # print(k_ref.shape)
    q_ref = q_ref.unsqueeze(0)[:, layer_ids]
    k_ref = k_ref.unsqueeze(0)[:, layer_ids]

    q_src = global_q[src_ids]
    k_src = global_k[src_ids]
    # print(q_src.shape)
    # print(k_src.shape)
    q_src = q_src[:, layer_ids]
    k_src = k_src[:, layer_ids]
    # print(q_src.shape)
    # print(k_src.shape)

    attn_map = q_ref @ q_src.transpose(-2, -1)
    # print(attn_map.shape)
    attn_map = rearrange(
        attn_map, "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok", n_h=n_h, n_w=n_w)
    # print(attn_map.shape)
    attn_map = attn_map.mean(dim=(2, 3)).std(dim=-1)
    attn_min = attn_map.min()
    attn_max = attn_map.max()

    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    # print(attn_map.shape)

    return attn_map


def extract_mean2_map(ref_id, global_q: torch.Tensor, global_k: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # mean q_ref_q_src 17-22
    window = torch.tensor([-6, -4, -2, 2, 4, 6])
    n_img = global_q.shape[0]
    img_h, img_w = images.shape[-2:]
    n_h, n_w = img_h // 14, img_w // 14

    src_ids = ref_id + window

    src_ids = src_ids[src_ids >= 0]
    src_ids = src_ids[src_ids < n_img]
    # print(src_ids)

    layer_ids = torch.arange(17, 22)

    q_ref = global_q[ref_id]
    k_ref = global_k[ref_id]
    # print(q_ref.shape)
    # print(k_ref.shape)
    q_ref = q_ref.unsqueeze(0)[:, layer_ids]
    k_ref = k_ref.unsqueeze(0)[:, layer_ids]

    q_src = global_q[src_ids]
    k_src = global_k[src_ids]
    # print(q_src.shape)
    # print(k_src.shape)
    q_src = q_src[:, layer_ids]
    k_src = k_src[:, layer_ids]
    # print(q_src.shape)
    # print(k_src.shape)

    attn_map = q_ref @ q_src.transpose(-2, -1)
    # print(attn_map.shape)
    attn_map = rearrange(
        attn_map, "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok", n_h=n_h, n_w=n_w)
    # print(attn_map.shape)
    attn_map = attn_map.mean(dim=(2, 3, 4))
    attn_min = attn_map.min()
    attn_max = attn_map.max()

    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    # print(attn_map.shape)

    return attn_map


def extract_mean3_map(ref_id, global_q: torch.Tensor, global_k: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # mean k_ref_k_src 0-1
    window = torch.tensor([-6, -4, -2, 2, 4, 6])
    n_img = global_q.shape[0]
    img_h, img_w = images.shape[-2:]
    n_h, n_w = img_h // 14, img_w // 14

    src_ids = ref_id + window

    src_ids = src_ids[src_ids >= 0]
    src_ids = src_ids[src_ids < n_img]
    # print(src_ids)

    layer_ids = torch.arange(0, 1)

    q_ref = global_q[ref_id]
    k_ref = global_k[ref_id]
    # print(q_ref.shape)
    # print(k_ref.shape)
    q_ref = q_ref.unsqueeze(0)[:, layer_ids]
    k_ref = k_ref.unsqueeze(0)[:, layer_ids]

    q_src = global_q[src_ids]
    k_src = global_k[src_ids]
    # print(q_src.shape)
    # print(k_src.shape)
    q_src = q_src[:, layer_ids]
    k_src = k_src[:, layer_ids]
    # print(q_src.shape)
    # print(k_src.shape)

    attn_map = k_ref @ k_src.transpose(-2, -1)
    # print(attn_map.shape)
    attn_map = rearrange(
        attn_map, "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok", n_h=n_h, n_w=n_w)
    # print(attn_map.shape)
    attn_map = attn_map.mean(dim=(2, 3, 4))
    attn_min = attn_map.min()
    attn_max = attn_map.max()

    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    # print(attn_map.shape)

    return attn_map


def extract_spacial_var3_map(ref_id, global_q: torch.Tensor, global_k: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    # spacial std q_ref_k_src 0-1
    window = torch.tensor([-6, -4, -2, 2, 4, 6])
    n_img = global_q.shape[0]
    img_h, img_w = images.shape[-2:]
    n_h, n_w = img_h // 14, img_w // 14

    src_ids = ref_id + window

    src_ids = src_ids[src_ids >= 0]
    src_ids = src_ids[src_ids < n_img]
    # print(src_ids)

    layer_ids = torch.arange(0, 1)

    q_ref = global_q[ref_id]
    k_ref = global_k[ref_id]
    # print(q_ref.shape)
    # print(k_ref.shape)
    q_ref = q_ref.unsqueeze(0)[:, layer_ids]
    k_ref = k_ref.unsqueeze(0)[:, layer_ids]

    q_src = global_q[src_ids]
    k_src = global_k[src_ids]
    # print(q_src.shape)
    # print(k_src.shape)
    q_src = q_src[:, layer_ids]
    k_src = k_src[:, layer_ids]
    # print(q_src.shape)
    # print(k_src.shape)

    attn_map = q_ref @ k_src.transpose(-2, -1)
    # print(attn_map.shape)
    attn_map = rearrange(
        attn_map, "n_img n_layer n_head (n_h n_w) n_tok -> n_h n_w (n_layer n_head) n_img n_tok", n_h=n_h, n_w=n_w)
    # print(attn_map.shape)
    attn_map = attn_map.mean(dim=(2, 3)).std(dim=-1)
    attn_min = attn_map.min()
    attn_max = attn_map.max()

    attn_map = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    # print(attn_map.shape)

    return attn_map


@torch.no_grad()
def extract_dyn_map(qk_dict: dict, images: torch.Tensor) -> torch.Tensor:
    dyn_maps = []
    n_img = images.shape[0]
    print(f"Extracting dynamic maps for {n_img} images")
    global_q = qk_dict["global_tok_q"].to("cuda")
    global_k = qk_dict["global_tok_k"].to("cuda")
    global_cam_q = qk_dict["global_cam_q"].to("cuda")
    for ref_id in tqdm(range(n_img)):
        mean1_map = extract_mean1_map(ref_id, global_q, global_k, images)
        mean2_map = extract_mean2_map(ref_id, global_q, global_k, images)
        mean3_map = extract_mean3_map(ref_id, global_q, global_k, images)
        var1_map = extract_spacial_var1_map(ref_id, global_q, global_k, images)
        var3_map = extract_spacial_var3_map(ref_id, global_q, global_k, images)

        dyn_map = (1 - mean1_map) * (1 - var1_map) * \
            (mean2_map) * (1 - mean3_map) * (var3_map)

        dyn_map_min = dyn_map.min()
        dyn_map_max = dyn_map.max()

        dyn_map = (dyn_map - dyn_map_min) / (dyn_map_max - dyn_map_min + 1e-6)
        dyn_maps.append(dyn_map)

    dyn_maps = torch.stack(dyn_maps)
    return dyn_maps.detach().cpu()


@torch.no_grad()
def batch_extract_dyn_map(qk_dict: dict, images: torch.Tensor) -> torch.Tensor:
    n_img, _, h_img, w_img = images.shape

    global_tok_q = qk_dict["global_tok_q"]
    global_tok_k = qk_dict["global_tok_k"]
    global_cam_q = qk_dict["global_cam_q"]

    n_batch = 50
    n_pad = 8
    dyn_maps = []
    for start_idx in range(0, n_img, n_batch):
        end_idx = min(start_idx + n_batch, n_img)
        b_start_idx = max(start_idx - n_pad, 0)
        b_end_idx = min(end_idx + n_pad, n_img)
        b_images = images[b_start_idx:b_end_idx]
        b_global_tok_q = global_tok_q[b_start_idx:b_end_idx]
        b_global_tok_k = global_tok_k[b_start_idx:b_end_idx]
        b_global_cam_q = global_cam_q[b_start_idx:b_end_idx]

        b_qk_dict = {
            "global_tok_q": b_global_tok_q,
            "global_tok_k": b_global_tok_k,
            "global_cam_q": b_global_cam_q,
        }
        b_dyn_maps = extract_dyn_map(b_qk_dict, b_images)
        b_dyn_mask_idx = torch.arange(
            start_idx - b_start_idx, end_idx - b_start_idx)
        dyn_map = b_dyn_maps[b_dyn_mask_idx]
        dyn_maps.append(dyn_map)

    dyn_maps = torch.cat(dyn_maps, dim=0)
    return dyn_maps


@torch.no_grad()
def cluster_attention_maps(feature, dynamic_map, n_clusters=64):
    """use KMeans to cluster the attention maps using feature

    Args:
        feature: encoder feature [B,H,W,C]
        dynamic_map: dynamic_map feature [B,H,W]
        n_clusters: number of clusters

    Returns:
        normalized_map: normalized cluster map [B,H,W]
        cluster_labels: reshaped cluster labels [B,H,W]
    """
    # data preprocessing
    B, H, W, C = feature.shape
    feature_np = feature.cpu().numpy()
    flattened_feature = feature_np.reshape(-1, C)

    # KMeans clustering
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(flattened_feature)

    # calculate the average dynamic score for each cluster
    dynamic_map_np = dynamic_map.cpu().numpy()
    flattened_dynamic = dynamic_map_np.reshape(-1)
    cluster_dynamic_scores = np.zeros(n_clusters, dtype=np.float32)
    if flattened_dynamic.size > 0:
        finite_vals = flattened_dynamic[np.isfinite(flattened_dynamic)]
        default_score = float(finite_vals.mean()) if finite_vals.size > 0 else 0.0
    else:
        default_score = 0.0
    for i in range(n_clusters):
        cluster_mask = (cluster_labels == i)
        if np.any(cluster_mask):
            cluster_vals = flattened_dynamic[cluster_mask]
            cluster_vals = cluster_vals[np.isfinite(cluster_vals)]
            if cluster_vals.size > 0:
                score = float(cluster_vals.mean())
                cluster_dynamic_scores[i] = score if np.isfinite(score) else default_score
            else:
                cluster_dynamic_scores[i] = default_score
        else:
            # Degenerate empty cluster, fallback to global finite mean.
            cluster_dynamic_scores[i] = default_score

    # map the cluster labels to the dynamic score
    cluster_map = cluster_dynamic_scores[cluster_labels]
    normalized_map = cluster_map.reshape(B, H, W)

    # reshape cluster_labels
    reshaped_labels = cluster_labels.reshape(B, H, W)

    # convert to torch tensor
    normalized_map = torch.from_numpy(normalized_map).float()
    cluster_labels = torch.from_numpy(reshaped_labels).long()

    # Guard against NaN/Inf values in degenerate scenes.
    normalized_map = torch.nan_to_num(normalized_map, nan=0.0, posinf=1.0, neginf=0.0)

    normalized_map_min = normalized_map.min(dim=1, keepdim=True)[
        0].min(dim=2, keepdim=True)[0]
    normalized_map_max = normalized_map.max(dim=1, keepdim=True)[
        0].max(dim=2, keepdim=True)[0]
    normalized_map = (normalized_map - normalized_map_min) / \
        (normalized_map_max - normalized_map_min + 1e-6)

    return normalized_map, cluster_labels


def adaptive_multiotsu_variance(img, verbose=False):
    """adaptive multi-threshold Otsu algorithm based on inter-class variance maximization

    Args:
        img: input image array
        verbose: whether to print detailed information

    Returns:
        tuple: (best threshold, best number of classes)
    """
    img = np.asarray(img)
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    if img.size == 0:
        return 0.0
    if not np.isfinite(img).all():
        mean_val = np.nanmean(img)
        return float(mean_val) if np.isfinite(mean_val) else 0.0

    img_min = float(np.min(img))
    img_max = float(np.max(img))
    if img_max <= img_min:
        # Flat map: no separable classes.
        return img_min

    max_classes = 4
    best_score = -float('inf')
    best_threshold = None
    best_n_classes = None
    scores = {}

    for n_classes in range(2, max_classes + 1):
        try:
            thresholds = threshold_multiotsu(img, classes=n_classes)
        except ValueError:
            # Degenerate histogram on short chunks; try other class counts.
            continue

        regions = np.digitize(img, bins=thresholds)
        var_between = np.var([img[regions == i].mean()
                             for i in range(n_classes)])

        score = var_between / np.sqrt(n_classes)
        scores[n_classes] = score

        if score > best_score:
            best_score = score
            best_threshold = thresholds[-1]
            best_n_classes = n_classes

    if verbose:
        print("number of classes score:")
        for n_classes, score in scores.items():
            print(f"number of classes {n_classes}: score {score:.4f}" +
                  (" (best)" if n_classes == best_n_classes else ""))
        print(f"final selected number of classes: {best_n_classes}")

    if best_threshold is None or not np.isfinite(best_threshold):
        return float(np.nanmedian(img))

    return float(best_threshold)

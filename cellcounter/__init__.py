# cellcounter/__init__.py

import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cellpose import core, denoise, utils, io, models, metrics
import gc

def mahotas_label(image):
    """
    takes a grayscale numpy array (.tiff file), returns (labeled_array, cell_count)
    
    usage:
        labels, count = cellcounter.mahotas_label(image)    
    """
    img = image.copy()
    if img.ndim == 3:
        img = img[:, :, 0]
    img = img.astype(np.float64)

    background = mh.gaussian_filter(img, 50) # sigma=50 to detect only background
    corrected = img - background
    corrected = corrected - corrected.min()

    smoothed = mh.gaussian_filter(corrected, 2)

    T = mh.thresholding.otsu(smoothed.astype(np.uint8))
    threshold = smoothed > T

    dist = mh.distance(threshold)
    dist = mh.gaussian_filter(dist.astype(np.float64), 4)
    rmax = mh.regmax(dist)
    seeds, _ = mh.label(rmax)

    labeled = mh.cwatershed(dist.max() - dist, seeds)
    labeled = labeled * threshold
    
    labeled, cell_count = mh.labeled.filter_labeled(
        labeled,
        remove_bordering=True,
        min_size=30,
        max_size=5000
    )

    labeled, cell_count = mh.labeled.relabel(labeled)

    return labeled, cell_count

def check_gpu():
    """
    checks for available resources.
    
    usage:
        gpu = check_gpu() # true or false
    """
    use_GPU = core.use_gpu()
    tf = [False, True]

    return tf[use_GPU]

def cellpose3_label(image):
    """
    takes a grayscale numpy array (.tiff file), returns (labeled_array, cell_count)
    
    usage:
        labels, count = cellcounter.cellpose3_label(image)
    """
    io.logger_setup()

    model = denoise.CellposeDenoiseModel(
        gpu=check_gpu(),
        model_type="cyto3",
        restore_type="denoise_cyto3"
    )
    masks, flows, styles, imgs_dn = model.eval([image], diameter=None, channels=[0,0])
    
    mask = masks[0] # (segmentation of one image)
    cell_count = mask.max()

    
    return mask, cell_count

def cellpose3_label_tiled(image, tile_size=256, overlap=64, iou_threshold=0.3):
    """
    takes a grayscale numpy array (.tiff file), slices it into small 'patches', segments it, and returs (labeled_array, cell_count)

    usage:
        labels, count = cellcounter.cellpose3_label_tiled(image)
    """
    img = image.copy()
    if img.ndim == 3:
        img = img[:, :, 0]

    h, w   = img.shape
    step   = tile_size - overlap
    canvas = np.zeros((h, w), dtype=np.int64)
    global_label_offset = 0

    model = denoise.CellposeDenoiseModel(
        gpu=check_gpu(),
        model_type="cyto3",
        restore_type="denoise_cyto3"
    )

    """
    generate tile starts that cover the full image, prevents incomplete segmentation due to input image size
    """
    def get_starts(length):
        starts = list(range(0, length - tile_size, step))
        if not starts or starts[-1] + tile_size < length:
            starts.append(length - tile_size)  # edge tile
        return starts

    for r in get_starts(h):
        for c in get_starts(w):
            tile      = img[r:r+tile_size, c:c+tile_size]
            masks, _, _, _ = model.eval([tile], diameter=None, channels=[0, 0])
            tile_mask = masks[0].astype(np.int64)

            # clean up old masks
            del masks
            gc.collect()

            if tile_mask.max() == 0:
                del tile_mask
                continue

            # capture how many cells found in tile before remapping
            n_cells_in_tile = int(tile_mask.max())

            # remap to globally unique IDs
            tile_mask[tile_mask > 0] += global_label_offset

            # increment by tile's own cell count, not the remapped max
            global_label_offset += n_cells_in_tile

            canvas[r:r+tile_size, c:c+tile_size] = np.where(
                tile_mask > 0,       # only write where cellpose found cells
                tile_mask,           # write the new label
                canvas[r:r+tile_size, c:c+tile_size]  # keep existing canvas otherwise
            )

            # free tile memory before next iteration
            del tile_mask
            gc.collect()

    # NMS deduplication (fixes duplicate segmentations due to patch overlap)
    canvas, cell_count = _nms_deduplicate(
        canvas, 
        iou_threshold=iou_threshold,
        tile_size=tile_size
        )

    return canvas, cell_count

def _compute_bboxes(labeled):
    """
    returns a dict of {label_id: (r_min, c_min, r_max, c_max)}
    for every labeled region.
    """
    bboxes = {}
    regions = mh.labeled.bbox(labeled, as_slice=False)
    # regions[i] = [r_min, r_max, c_min, c_max] for label i
    for label_id in range(1, labeled.max() + 1):
        if label_id < len(regions):
            r_min, r_max, c_min, c_max = regions[label_id]
            bboxes[label_id] = (r_min, c_min, r_max, c_max)
    return bboxes


def _iou(box_a, box_b):
    """
    computes Intersection over Union for two bounding boxes.
    Each box is (r_min, c_min, r_max, c_max).
    """
    r_min = max(box_a[0], box_b[0])
    c_min = max(box_a[1], box_b[1])
    r_max = min(box_a[2], box_b[2])
    c_max = min(box_a[3], box_b[3])

    intersection = max(0, r_max - r_min) * max(0, c_max - c_min)
    if intersection == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _nms_deduplicate(labeled, iou_threshold=0.3, tile_size=512):
    """
    removes duplicate cells introduced at tile borders using NMS.
    keeps the larger of any two regions with IoU above iou_threshold.

    returns a cleaned labeled array.
    """
    labeled = labeled.astype(np.int64)
    bboxes   = _compute_bboxes(labeled)
    label_ids = list(bboxes.keys())

    # compute pixel size of each region for scoring
    sizes = mh.labeled.labeled_size(labeled)

    to_remove = set()

    for i, id_a in enumerate(label_ids):
        if id_a in to_remove:
            continue

        box_a = bboxes[id_a]

        for id_b in label_ids[i+1:]:
            if id_b in to_remove:
                continue

            box_b = bboxes[id_b]

            # skip comparison if bboxes are one tile_size apart, as they cannot be duplicates if this is the case
            if abs(box_a[0] - box_b[0]) > tile_size:
                continue
            if abs(box_a[1] - box_b[1]) > tile_size:
                continue

            iou = _iou(box_a, box_b)
            if iou > iou_threshold:
                # discard the smaller region
                if sizes[id_a] >= sizes[id_b]:
                    to_remove.add(id_b)
                else:
                    to_remove.add(id_a)
                    break  # id_a is gone, move to next id_a

    # zero out removed regions
    cleaned = labeled.copy()
    for label_id in to_remove:
        cleaned[cleaned == label_id] = 0

    # relabel to close gaps
    cleaned, cell_count = mh.labeled.relabel(cleaned)
    return cleaned, int(cell_count)

def label(image, method='mahotas'):
    """
    takes a grayscale numpy array (.tiff file) and segmentation method, returns call to specified method on input image.
    
    usage:
        labels, count = cellcounter.label(image, method='cellpose3')
    """

    supported_methods = ['mahotas', 'cellpose3', 'cellpose3_tiled']

    if method == 'mahotas':
        return mahotas_label(image)
    elif method == 'cellpose3':
        return cellpose3_label(image)
    elif method == 'cellpose3_tiled':
        return cellpose3_label_tiled(image)
    else:
        raise ValueError(f"Unknown method {method}. The only methods of labelling currently supported are: " + supported_methods)

def display(image, labels, count):
    """
    displays the original image with the labeled overlay and cell count.
    
    usage:
        cellcounter.display(image, labels, count)
    """

    # normalize original image for displaying
    img = image.copy().astype(np.float64)
    if img.ndim == 3:
        img = img[:, :, 0]

    display_img = img - img.min()
    display_img = (display_img / display_img.max() * 255).astype(np.uint8)
    rgb = np.stack([display_img, display_img, display_img], axis=-1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, cmap='gray')

    if hasattr(utils, 'outlines_list'):
        try:
            outlines = utils.outlines_list(labels)
            for outline in outlines:
                ax.plot(outline[:,1], outline[:,0], color='yellow', linewidth=0.8)
        except Exception:
            _draw_filled_overlay(ax, rgb, labels)
    else:
        _draw_filled_overlay(ax, rgb, labels)
    
    ax.set_title(f"{count} cells", fontsize=18, fontweight='bold', pad=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def _draw_filled_overlay(ax, rgb, labels):
    """
    internal helper for display() to draw colored cell regions for mahotas labels
    """

    # colorize labeled regions
    cmap = plt.get_cmap("nipy_spectral")
    max_label = labels.max()
    if max_label > 0:
        colors = (cmap(labels.astype(float) / max_label)[:,:,:3]*255).astype(np.uint8)
        mask = labels > 0
        alpha = 0.55
        rgb[mask] = (rgb[mask] * (1 - alpha) + colors[mask] * alpha).astype(np.uint8)

    # plot display
    ax.imshow(rgb)  
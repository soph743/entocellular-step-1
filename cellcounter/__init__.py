# cellcounter/__init__.py

import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from cellpose import core, denoise, utils, io, models, metrics

def mahotas_label(image):
    """
    takes a grayscale numpy array (.tiff file), returns (labeled_array, cell_count)
    
    usage:
        labels, count = cellcounter.mahotas_label(image)
    """

    # normalize to 2D if needed
    img = image.copy()
    if img.ndim == 3:
        img = img[:, :, 0]
    img = img.astype(np.float64)

    # mahotas segmentation pipeline
    smoothed = mh.gaussian_filter(img, 4)
    threshold = smoothed > smoothed.mean()
    labeled, cell_count = mh.label(threshold)
    labeled, cell_count = mh.labeled.filter_labeled(labeled, remove_bordering=True)

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

def label(image, method='mahotas'):
    """
    takes a grayscale numpy array (.tiff file) and segmentation method, returns call to specified method on input image.
    
    usage:
        labels, count = cellcounter.label(image, method='cellpose3')
    """

    supported_methods = ['mahotas', 'cellpose3']

    if method == 'mahotas':
        return mahotas_label(image)
    elif method == 'cellpose3':
        return cellpose3_label(image)
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
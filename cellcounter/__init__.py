# cellcounter/__init__.py

import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def mahotas_label(image):
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

def label(image, method='mahotas'):
    """
    takes a grayscale numpy array (.tiff file), returns (labeled_array, cell_count)
    
    usage:
        labels, count = cellcounter.label(image)
    """

    supported_methods = ['mahotas']

    if method not in supported_methods:
        raise ValueError(f"Unknown method {method}. The only methods of labelling currently supported are: " + supported_methods)
    
    return mahotas_label(image)
    

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

    # colorize labeled regions
    cmap = plt.get_cmap("Purples")
    max_label = labels.max()
    if max_label > 0:
        colors = (cmap(labels.astype(float) / max_label)[:,:,:3]*255).astype(np.uint8)
        mask = labels > 0
        alpha = 0.9
        rgb[mask] = (rgb[mask] * (1 - alpha) + colors[mask] * alpha).astype(np.uint8)

    # plot display
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(rgb)
    ax.set_title(f"Cells Detected: {count}", fontsize=18, fontweight='bold', pad=16)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
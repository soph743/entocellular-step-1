from flask import Flask, request, jsonify, render_template
import mahotas as mh
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

app = Flask(__name__)

@app.route("/") # home page
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"]) # routing for when an image is uploaded
def analyze():
    # 1. read the uploaded .tiff file
    file = request.files["image"]
    img_array = tifffile.imread(file) # converts image file to array

    # 2. normalize to 2D grayscale, if needed
    if img_array.ndim == 3:
        img_array = img_array[:, :, 0]
    img_array = img_array.astype(np.float64)

    # 3. mahotas cell segmentation pipeline
    smoothed = mh.gaussian_filter(img_array, 4) # smooths the image to reduce noise
    threshold = smoothed > smoothed.mean() 
    # from mahotas docs: "without the Gaussian filter, the resulting thresholded image has
    # very noisy edges"
    labeled, cell_count = mh.label(threshold) # label the connected regions

    # optional: filter out tiny noise regions and border-touching cells
    labeled, cell_count = mh.labeled.filter_labeled(
        labeled, remove_bordering=True # no min_size argument since cells are very small
    )

    # 4. build a colorized overlay image, normalize original for display
    display = img_array - img_array.min()
    display = (display / display.max() * 255).astype(np.uint8)
    rgb = np.stack([display, display, display], axis=-1) # convert gscl to rgb

    # use matplotlib colormap to color each labeled region
    cmap = plt.get_cmap("Purples")
    max_label = labeled.max()
    
    # if the largest labeled cell is > 0 then normalize all the labeled regions and convert to binary [0, 1]
    if max_label > 0:
        colors = (cmap(labeled.astype(float) / max_label)[:, :, :3] * 255).astype(np.uint8)
        mask = labeled > 0 # get foreground of labeled regions
        alpha = 0.9
        rgb[mask] = (rgb[mask] * (1 - alpha) + colors[mask] * alpha).astype(np.uint8) # build overlay using cmap and mask

    # 5. encode result as base64 PNG
    pil_img = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "cell_count": int(cell_count),
        "overlay_image": f"data:image/png;base64,{encoded}"
    })

# boiler plate
if __name__ == "__main__":
    app.run(debug=True)
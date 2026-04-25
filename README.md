# entocellular-step-1
A prototype for a CLI and web tool for counting cells in whole-slide-images for Entocellular's ML project

# 1. web tool for counting cells
a tool for just dropping .tiff images and getting the number of cells in the image. meant for visualization mainly.

## 1a. how it works
`[Browser: drag & drop .tiff]
        ↓  POST /analyze (multipart form)
[Flask Backend]
  → Read .tiff with tifffile
  → Gaussian filter (mahotas)
  → Threshold (mahotas)
  → mh.label() → nr_objects
  → Colorize labeled image
  → Encode as base64 PNG
        ↓  JSON { cell_count, overlay_image }
[Browser: display overlay + count]`

## 1b. project structure

`cell-counter-web/
├── app.py
├── requirements.txt
└── templates/
    └── index.html`
    
# 2. cli tool (starter code -- to be built out in future work)
eventually this could be expanded to support various methods for counting cells (not just limited to those provided in Mahotas). 

currently, this tool can be used to run a script `generate_counts.py` that returns the count of cells in an image.

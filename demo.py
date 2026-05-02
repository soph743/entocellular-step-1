import tifffile
import cellcounter

image = tifffile.imread('sample-data/Well_C2_Xmm0.987_Ymm-0.999_Ch1_1um.tiff')

labels, count = cellcounter.label(image, method='mahotas')
print(f"Found {count} cells using mahotas.")

#cellcounter.display(image, labels, count)

labels, count = cellcounter.label(image, method='cellpose3')
print(f"Found {count} cells using cellpose3.")
cellcounter.display(image, labels, count)
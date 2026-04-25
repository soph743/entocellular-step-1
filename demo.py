import tifffile
import cellcounter

image = tifffile.imread('sample-data/Well_C2_Xmm0.987_Ymm-0.999_Ch1_1um.tiff')

labels, count = cellcounter.label(image, method='mahotas')
print(f"Found {count} cells.")

cellcounter.display(image, labels, count)
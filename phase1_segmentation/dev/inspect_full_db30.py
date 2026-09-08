import rasterio
import numpy as np

path = "/mnt/e/SIH project/satellite data/predictions_full_db30/wakashio_probability_db30.tif"

with rasterio.open(path) as src:
    data = src.read(1).astype(np.float32)

    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("CRS:", src.crs)
    print("Pixel size:", src.res)

    print("\nStatistics:")
    print("Min:", data.min())
    print("Max:", data.max())
    print("Mean:", data.mean())
    print("Median:", np.median(data))

    for p in [1, 5, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9]:
        print(f"P{p}:", np.percentile(data, p))

    print("\nThreshold 0.20 equivalent in uint8:", 0.20 * 255)
    print(
        "Fraction >= 0.20:",
        np.mean(data >= 0.20 * 255)
    )
from tifffile import TiffFile

def extract_tiff_metadata(filepath):
    """
    Extract metadata from a standard TIFF file, including physical pixel sizes, bit depth, and shape.
    """
    with TiffFile(filepath) as tif:
        metadata = {}
        page = tif.pages[0]  # Use the first page for metadata
        
        # Extract X and Y resolution (DPI) and convert to µm/pixel
        x_res = page.tags.get('XResolution')
        y_res = page.tags.get('YResolution')
        
        metadata["PhysicalSizeX"] = 1000 / (x_res.value[0] / x_res.value[1]) if x_res else 1.0
        metadata["PhysicalSizeXUnit"] = "µm"
        metadata["PhysicalSizeY"] = 1000 / (y_res.value[0] / y_res.value[1]) if y_res else 1.0
        metadata["PhysicalSizeYUnit"] = "µm"

        # Extract bit depth (significant bits)
        metadata["SignificantBits"] = page.tags.get('BitsPerSample').value if page.tags.get('BitsPerSample') else 16

        # Extract data type
        metadata["Type"] = tif.series[0].dtype.name if tif.series else "unknown"

        # Extract image shape
        metadata["Shape"] = tif.series[0].shape if tif.series else (1, 1, 1, 1, 1)

        return metadata

if __name__ == "__main__":
    filepath = "resources/example_images/2C_3D_testTIF.tif"  # Update the path
    metadata = extract_tiff_metadata(filepath)
    print("\nExtracted Metadata:")
    for key, value in metadata.items():
        print(f"{key}: {value}")

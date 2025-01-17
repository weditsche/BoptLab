import os
import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt

from skimage import feature
from skimage.filters import difference_of_gaussians

def load_tiff_stack(filepath):
    """
    Load a TIFF stack using tifffile.
    Returns a numpy array.

    Possible shapes:
      - (T, Y, X) if 2D time series
      - (T, Z, Y, X) if 3D time series (time + z-stack)
      - other permutations depending on your export method

    Adjust if your data’s axes are in a different order.
    """
    with tifffile.TiffFile(filepath) as tif:
        data = tif.asarray()
    return data

def detect_bursts_2d(image_2d, sigma_small=1, sigma_large=3, threshold_rel=0.2):
    """
    Detect dot-like bursts in a 2D image using:
      1) Difference-of-Gaussian (DoG) filtering
      2) Local peak detection (peak_local_max)

    Parameters
    ----------
    image_2d : np.ndarray
        2D grayscale image.
    sigma_small : float
        Smaller sigma for DoG.
    sigma_large : float
        Larger sigma for DoG.
    threshold_rel : float
        Relative threshold (fraction of max) for peak detection in the DoG response.

    Returns
    -------
    coords : np.ndarray
        (N, 2) array of (row, col) coordinates of detected spots.
    """
    # Apply Difference-of-Gaussian
    dog = difference_of_gaussians(
        image_2d, 
        sigma_small, 
        sigma_large
    )

    # Threshold is relative to the maximum value in the DoG image
    dog_max = np.max(dog)
    absolute_threshold = threshold_rel * dog_max if dog_max != 0 else 0

    coords = feature.peak_local_max(
        dog, 
        min_distance=2,           # Adjust min_distance to avoid merging close spots
        threshold_abs=absolute_threshold
    )
    return coords

def analyze_time_series(data, sigma_small=1, sigma_large=3, threshold_rel=0.2):
    """
    Detect bursts in a time-series data. 
    This function assumes data is either:
      - 3D (T, Y, X)  (i.e., MIP already done)
      - 4D (T, Z, Y, X), in which case you might want to
        reduce it or handle each slice individually.

    Parameters
    ----------
    data : np.ndarray
        The time series data. 
        Shape could be (T, Y, X) or (T, Z, Y, X).
    sigma_small, sigma_large : float
        Difference-of-Gaussian parameters.
    threshold_rel : float
        Threshold for local maxima.

    Returns
    -------
    burst_info : dict
        Dictionary mapping time index -> list of (z, row, col) or (row, col)
        depending on data shape.
    """
    burst_info = {}
    
    if data.ndim == 3:
        # (T, Y, X)
        T, Y, X = data.shape
        for t in range(T):
            frame_2d = data[t]  # shape (Y, X)
            coords = detect_bursts_2d(
                frame_2d, 
                sigma_small=sigma_small, 
                sigma_large=sigma_large, 
                threshold_rel=threshold_rel
            )
            burst_info[t] = coords  # (row, col) for each detected spot

    elif data.ndim == 4:
        # (T, Z, Y, X)
        T, Z, Y, X = data.shape
        for t in range(T):
            # Option 1: detect bursts in each Z slice and combine
            # Option 2: you might do your own 3D spot detection,
            #           but here we do 2D per slice for demonstration.
            coords_this_t = []
            for z in range(Z):
                frame_2d = data[t, z]  # shape (Y, X)
                coords = detect_bursts_2d(
                    frame_2d, 
                    sigma_small=sigma_small, 
                    sigma_large=sigma_large, 
                    threshold_rel=threshold_rel
                )
                # Tag each coordinate with the slice index
                coords_z_tagged = [(z, r, c) for (r, c) in coords]
                coords_this_t.extend(coords_z_tagged)
            
            burst_info[t] = coords_this_t
    else:
        raise ValueError(f"Unsupported data shape {data.shape}. Expected 3D or 4D.")
    
    return burst_info

def save_burst_info_to_csv(burst_info, output_csv):
    """
    Saves burst coordinates to CSV in the format:
        time,z,row,col
    or
        time,row,col
    depending on whether data was 4D or 3D.
    """
    # Check if we have any (z, r, c) or just (r, c)
    # We'll inspect the first timepoint.
    first_time = sorted(burst_info.keys())[0]
    sample_coords = burst_info[first_time]
    
    # Decide on the CSV header
    # If sample_coords is empty, we can’t guess. So handle that edge case:
    if len(sample_coords) == 0:
        # We'll assume 2D
        is_4d = False
    else:
        # If the first coordinate is a tuple of length 3, we have (z, r, c).
        is_4d = len(sample_coords[0]) == 3

    with open(output_csv, "w") as fh:
        if is_4d:
            fh.write("time,z,row,col\n")
        else:
            fh.write("time,row,col\n")
        
        for t in sorted(burst_info.keys()):
            coords_t = burst_info[t]
            for coord in coords_t:
                if is_4d:
                    z, r, c = coord
                    fh.write(f"{t},{z},{r},{c}\n")
                else:
                    r, c = coord
                    fh.write(f"{t},{r},{c}\n")

    print(f"Saved burst coordinates to {output_csv}")


def main():
    """
    Main workflow:
      1) Find all .tif / .tiff files in a folder
      2) Load them
      3) Detect bursts
      4) Save results (coordinates) to CSV
      5) (Optional) visualize a random time frame
    """

    # Folder containing your pre-processed TIFF files
    input_folder = "./data/mgarcia"  # <-- change this
    output_folder = "./outputs"      # <-- change this
    os.makedirs(output_folder, exist_ok=True)

    # Gather all tif/tiff files in this folder
    file_list = glob.glob(os.path.join(input_folder, "*.tif"))
    file_list += glob.glob(os.path.join(input_folder, "*.tiff"))
    file_list.sort()

    if not file_list:
        print("No TIFF files found in the specified folder.")
        return

    # Parameters for burst detection
    sigma_small = 1
    sigma_large = 3
    threshold_rel = 0.2   # Adjust as needed

    for i, filepath in enumerate(file_list, start=1):
        print(f"({i}/{len(file_list)}) Analyzing: {filepath}")
        
        # 1) Load data
        data = load_tiff_stack(filepath)

        # 2) Detect bursts
        burst_info = analyze_time_series(
            data, 
            sigma_small=sigma_small, 
            sigma_large=sigma_large, 
            threshold_rel=threshold_rel
        )

        # 3) Save to CSV
        # Make an output CSV filename based on input filename
        filename_no_ext = os.path.splitext(os.path.basename(filepath))[0]
        output_csv = os.path.join(output_folder, f"{filename_no_ext}_bursts.csv")
        save_burst_info_to_csv(burst_info, output_csv)

        # 4) (Optional) visualize a random timepoint for QC
        #    Only if data is 3D (T, Y, X). 
        #    If data is 4D, you'd do a separate approach (picking a Z slice to display).
        if data.ndim == 3:
            T, Y, X = data.shape
            rand_t = np.random.randint(0, T)
            frame_2d = data[rand_t]
            coords_t = burst_info[rand_t]  # (r, c) for each spot
            
            plt.figure(figsize=(6, 5))
            plt.title(f"{filename_no_ext} - Time {rand_t} (Detected Bursts)")
            plt.imshow(frame_2d, cmap='gray')
            if len(coords_t) > 0:
                rr, cc = zip(*coords_t)
                plt.plot(cc, rr, 'ro', markersize=2)
            plt.show()

if __name__ == "__main__":
    main()

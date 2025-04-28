import os
import numpy as np
from pathlib import Path
import test_segy_io

# --- Configuration ---
slice_dir = Path("./inpainted_slices")
output_dir = Path("./reconstructed")
output_segy_filename = "infilled_volume_final_norm.sgy" # Keep consistent name
# SEGY Header Info
client_name = "Inpainting Result"
sample_interval_us = 2000
inline_step = 1
crossline_step = 1
origin_y = 0.0
origin_x = 0.0
spacing_y = 25.0
spacing_x = 25.0
coord_scalar = -100
projection_desc = "UNKNOWN PROJECTION"
provider_name = "MOMACMO LIMITED"
x_axis_azimuth = 90.0
# --- End Configuration ---

def reconstruct_volume_from_slices(slice_dir, output_dir, output_filename):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_segy_path = output_dir / output_filename

    slice_files = []
    positions = []
    min_pos = float('inf')
    max_pos = float('-inf')
    try:
        for item in slice_dir.glob("slice_z_*.npy"):
             filename_part = item.stem.split('_')[-1]
             if filename_part.isdigit():
                 pos = int(filename_part)
                 slice_files.append((pos, item))
                 positions.append(pos)
                 min_pos = min(min_pos, pos)
                 max_pos = max(max_pos, pos)
             else:
                 print(f"Warning: Skipping file with non-numeric suffix: {item.name}")

        if not slice_files:
             raise FileNotFoundError(f"No valid 'slice_z_*.npy' files found in {slice_dir.resolve()}")

        slice_files.sort()
        print(f"Found {len(slice_files)} slice files. Positions: {min_pos} to {max_pos}")

    except Exception as e:
        print(f"Error finding or parsing slice files: {e}")
        return

    slices = []
    for pos, slice_file in slice_files:
        try:
            slice_data = np.load(slice_file)
            slices.append(slice_data)
        except Exception as e:
            print(f"Error loading slice {slice_file}: {e}")
            return

    if not slices:
        print("Error: No slices were loaded.")
        return
    first_shape = slices[0].shape
    if not all(s.shape == first_shape for s in slices):
        print("Error: Not all slices have the same dimensions.")
        return
    print(f"All slices loaded successfully with shape: {first_shape}")

    inline_start_val = max(1, min_pos)
    crossline_start_val = max(1, min_pos)
    print(f"Using inline_start={inline_start_val}, crossline_start={crossline_start_val}.")

    try:
        volume_3d = np.stack(slices, axis=-1).astype(np.float32)
        print(f"Reconstructed volume shape: {volume_3d.shape}")

        # Robust Data Normalization
        print("Applying normalization...")
        if np.isnan(volume_3d).any() or np.isinf(volume_3d).any():
            print("Warning: NaN/Inf found. Replacing with 0.")
            volume_3d = np.nan_to_num(volume_3d, nan=0.0, posinf=0.0, neginf=0.0)
        try:
             p2, p98 = np.percentile(volume_3d, (2, 98))
             # print(f"  P2={p2:.4f}, P98={p98:.4f}") # Optional: uncomment for debugging
             volume_3d = np.clip(volume_3d, p2, p98)
             range_val = p98 - p2
             if not np.isclose(range_val, 0):
                 volume_3d = (volume_3d - p2) / range_val # Scale to [0, 1]
                 volume_3d = (volume_3d * 2.0) - 1.0 # Scale to [-1, 1]
             else:
                 volume_3d.fill(0.0) # Avoid division by zero, set to mid-range (0 for [0,1]) -> -1
                 print("Warning: P2 and P98 are close. Scaled data might be uniform.")
        except Exception as e:
             print(f"Warning: Error during normalization: {e}. Skipping.")
        print("Normalization complete.")

    except Exception as e:
        print(f"Error stacking or normalizing data: {e}")
        return

    print(f"Writing SEGY file: {output_segy_path.resolve()}")
    try:
        test_segy_io.write_3d_numpy_to_segy_with_coordinates(
            filename=str(output_segy_path),
            data=volume_3d,
            client=client_name,
            sample_interval_us=sample_interval_us,
            inline_start=inline_start_val,
            crossline_start=crossline_start_val,
            inline_step=inline_step,
            crossline_step=crossline_step,
            origin=(origin_y, origin_x),
            spacing=(spacing_y, spacing_x),
            x_axis_azimuth=x_axis_azimuth,
            coord_scalar=coord_scalar,
            projection=projection_desc,
            provider=provider_name
        )
        print("SEGY file writing complete.")
    except Exception as e:
        print(f"Error writing SEGY file: {e}")

if __name__ == "__main__":
    reconstruct_volume_from_slices(slice_dir, output_dir, output_segy_filename)
    print("\nScript finished.")


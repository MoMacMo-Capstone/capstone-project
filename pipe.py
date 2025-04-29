import torch
import torch.nn as nn
import numpy as np
import draw_mask
from model import CombinedGenerator, device, resolution
import os
from pathlib import Path
import test_segy_io
import reconstruct_segy
# Select the desired model checkpoint
# G = CombinedGenerator("trained_128x128_b54a594.ckpt").to(device)
# G = CombinedGenerator("trained_64x64_cbbfce7_div1.ckpt").to(device)
G = CombinedGenerator("trained_64x64_305ee13_div1.ckpt").to(device) # Added map_location for robustness

# --- SEGY Output Configuration ---
# (Moved from reconstruct_segy.py)
output_dir = Path("./reconstructed") # Directory for the final SEGY
output_segy_filename = "inpainted_direct_output.sgy" # Name for the output SEGY
# SEGY Header Info
client_name = "Inpainting Result (Direct)"
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
# --- End SEGY Output Configuration ---

def center_of_mass_and_rectangle(mask, rect_size):
    if mask.ndim != 2:
        raise ValueError("Input mask must be a 2D array.")

    height, width = mask.shape
    rect_width, rect_height = rect_size

    indices = np.argwhere(mask)
    if indices.size == 0:
        # Default to center if mask is empty or has no selection
        print("Warning: Mask is empty or has no selection. Defaulting rectangle to center.")
        center_y, center_x = height // 2, width // 2
    else:
        center_y, center_x = np.mean(indices, axis=0)

    center_x = int(round(center_x))
    center_y = int(round(center_y))

    half_width = rect_width // 2
    half_height = rect_height // 2

    left = max(center_x - half_width, 0)
    right = min(center_x + half_width + (rect_width % 2), width)
    top = max(center_y - half_height, 0)
    bottom = min(center_y + half_height + (rect_height % 2), height)

    # Adjust rectangle if it exceeds boundaries due to centering
    if right - left < rect_width:
        if left == 0:
            right = min(rect_width, width)
        elif right == width:
            left = max(width - rect_width, 0)

    if bottom - top < rect_height:
        if top == 0:
            bottom = min(rect_height, height)
        elif bottom == height:
            top = max(height - rect_height, 0)

    # Final check to ensure the extracted rectangle is exactly rect_size
    # This might crop slightly differently if the mask center is near an edge
    final_left = left
    final_top = top
    final_right = min(left + rect_width, width)
    final_bottom = min(top + rect_height, height)
    # Adjust left/top if right/bottom hit the boundary first
    final_left = max(final_right - rect_width, 0)
    final_top = max(final_bottom - rect_height, 0)


    return (final_top, final_left, final_bottom, final_right)

def match_shape(tensor, target):
    _, _, h, w = tensor.shape
    _, _, H, W = target.shape
    dh, dw = H - h, W - w
    if dh > 0 or dw > 0:
        tensor = nn.functional.pad(tensor, [0, dw, 0, dh])
    elif dh < 0 or dw < 0:
        tensor = tensor[:, :, :H, :W]
    return tensor

def infill_and_display(model, volume, mask):
    """
    Performs inpainting on the volume based on the mask and returns the result.
    Optionally displays intermediate results if needed.
    """
    model.eval()
    infilled_volume = volume.copy() # Start with original volume
    # Initialize arrays for optional display (can be removed if display not needed)
    noise_volume = np.zeros_like(volume)
    mean_volume = volume.copy()
    stdev_volume = np.zeros_like(volume)

    mask_applied = np.any(mask)
    if mask_applied:
        try:
            top, left, bottom, right = center_of_mass_and_rectangle(mask, resolution)
            print(f"Calculated bounding box for inpainting: top={top}, left={left}, bottom={bottom}, right={right}")
            if (bottom - top) != resolution[0] or (right - left) != resolution[1]:
                 print(f"Warning: Extracted patch size ({bottom-top}x{right-left}) doesn't match model resolution ({resolution[0]}x{resolution[1]}). Check mask or centering logic.")
                 # Proceed anyway, model might handle slight mismatches depending on padding/cropping
        except ValueError as e:
            print(f"Error calculating bounding box: {e}. Skipping inpainting.")
            mask_applied = False # Treat as if no mask was applied

    if not mask_applied:
        print("No mask applied or bounding box error. Returning original volume.")
        return volume # Return the original volume if no inpainting happens

    # Perform inpainting slice by slice
    print("Starting inpainting process...")
    with torch.no_grad():
        for i in range(volume.shape[2]):
            slice_input = volume[:, :, i]

            # Extract the patch to inpaint
            cropped_input = slice_input[top:bottom, left:right]
            cropped_mask = mask[top:bottom, left:right] # Use the 2D mask crop for all slices

            # Ensure patch is exactly the expected resolution (pad if needed, crop if needed)
            # This assumes CombinedGenerator expects exactly resolution size input
            h_crop, w_crop = cropped_input.shape
            h_res, w_res = resolution
            pad_h = max(0, h_res - h_crop)
            pad_w = max(0, w_res - w_crop)
            # Simple padding - more sophisticated might be needed if model is sensitive
            if pad_h > 0 or pad_w > 0:
                 cropped_input = np.pad(cropped_input, ((0, pad_h), (0, pad_w)), mode='constant')
                 cropped_mask = np.pad(cropped_mask, ((0, pad_h), (0, pad_w)), mode='constant')
            # Crop if extracted patch was too large (shouldn't happen with logic above)
            cropped_input = cropped_input[:h_res, :w_res]
            cropped_mask = cropped_mask[:h_res, :w_res]


            input_tensor = torch.tensor(cropped_input, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.tensor(cropped_mask, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)

            # Run model
            infill, noise, mean, stdev = model.forward_with_intermediate(input_tensor, mask_tensor)
            infill_np = infill.squeeze().cpu().numpy()
            noise_np = noise.squeeze().cpu().numpy()
            mean_np = mean.squeeze().cpu().numpy()
            stdev_np = stdev.squeeze().cpu().numpy()

            # Place the potentially size-adjusted inpainted result back
            # Ensure we only place back where the mask was true
            patch_h, patch_w = infill_np.shape
            h_target = min(patch_h, bottom-top)
            w_target = min(patch_w, right-left)

            target_patch = infilled_volume[top:top+h_target, left:left+w_target, i]
            source_patch = infill_np[:h_target, :w_target]
            mask_patch = cropped_mask[:h_target, :w_target]

            target_patch[mask_patch] = source_patch[mask_patch]
            infilled_volume[top:top+h_target, left:left+w_target, i] = target_patch


            # Store intermediate results for optional display
            # Adjust index slicing similar to infill placing
            noise_volume[top:top+h_target, left:left+w_target, i] = noise_np[:h_target, :w_target]
            mean_volume[top:top+h_target, left:left+w_target, i] = mean_np[:h_target, :w_target]
            stdev_volume[top:top+h_target, left:left+w_target, i] = stdev_np[:h_target, :w_target]


            if (i + 1) % 50 == 0 or i == volume.shape[2] - 1: # Print progress less often
                print(f"Processed slice {i + 1} of {volume.shape[2]}")

    print("Inpainting complete.")
    # Display results (optional)
    print("Displaying results (optional)...")
    try:
        draw_mask.show_volume_with_slider(volume, mask, infilled_volume, noise_volume, mean_volume, stdev_volume)
        print("Close the display window to finish.")
    except Exception as e:
        print(f"Could not display results: {e}")

    # Return the final 3D volume
    return infilled_volume


def load_data_from_segy(segy_filename):
    print(f"Attempting to load SEGY file: {segy_filename}")
    try:
        segy_data_3d, _, _, _ = test_segy_io.read_segy_to_3d_numpy(segy_filename)
        print(f"Successfully loaded SEGY data with shape: {segy_data_3d.shape}")
        segy_data_4d = np.expand_dims(segy_data_3d, axis=0)
        print(f"Reshaped SEGY data to 4D: {segy_data_4d.shape}")
        return segy_data_4d
    except FileNotFoundError:
        print(f"Error: SEGY file not found at {segy_filename}")
        return None
    except Exception as e:
        print(f"Error loading SEGY file '{segy_filename}': {e}")
        return None


if __name__ == "__main__":
    model = G
    data = None
    input_inline_vals = None # To store original geometry if loaded from SEGY
    input_crossline_vals = None

    # --- Load Data ---
    while data is None:
        load_choice = input("Load from (1) Default BIN file or (2) SEGY file? Enter 1 or 2: ")

        if load_choice == '1':
            bin_filename = "SegActi-45x201x201x614.bin"
            print(f"Loading data from default BIN file: {bin_filename}")
            try:
                with open(bin_filename, "rb") as f:
                    w, x, y, z = 45, 201, 201, 614
                    bytes_to_read = w * x * y * z * 4
                    file_bytes = f.read(bytes_to_read)
                    if len(file_bytes) != bytes_to_read:
                         raise ValueError(f"Expected {bytes_to_read} bytes, but found {len(file_bytes)}")
                    data = np.frombuffer(file_bytes, dtype="f4").reshape(w, x, y, z)
                print(f"Loaded BIN data with shape: {data.shape}")
                # For BIN, we don't know original inline/crossline numbers, assume starting at 1
                input_inline_vals = np.arange(1, data.shape[1] + 1)
                input_crossline_vals = np.arange(1, data.shape[2] + 1)
            except FileNotFoundError:
                print(f"Error: Default BIN file '{bin_filename}' not found.")
            except Exception as e:
                print(f"Error reading BIN file '{bin_filename}': {e}")

        elif load_choice == '2':
            segy_filename = input("Enter the path to the SEGY file: ")
            # Modify load_data_from_segy slightly to return geometry if available
            print(f"Attempting to load SEGY file: {segy_filename}")
            try:
                segy_data_3d, il_vals, xl_vals, _ = test_segy_io.read_segy_to_3d_numpy(segy_filename)
                print(f"Successfully loaded SEGY data with shape: {segy_data_3d.shape}")
                data = np.expand_dims(segy_data_3d, axis=0) # Make 4D
                print(f"Reshaped SEGY data to 4D: {data.shape}")
                input_inline_vals = il_vals
                input_crossline_vals = xl_vals
            except FileNotFoundError:
                print(f"Error: SEGY file not found at {segy_filename}")
                data = None
            except Exception as e:
                print(f"Error loading SEGY file '{segy_filename}': {e}")
                data = None
        else:
            print("Invalid choice. Please enter 1 or 2.")

    # --- Proceed if data loaded successfully ---
    if data is not None:
        print("-" * 20)
        # 1. Choose Volume
        volume = draw_mask.choose_volume(data) # Selects the 3D volume
        print(f"Selected volume shape for processing: {volume.shape}")

        # 2. Draw Mask
        print("Please draw the mask for inpainting...")
        mask = draw_mask.make_drawn_mask_2d(volume) # Get the 2D mask

        # 3. Inpaint Volume
        print("Mask created. Starting inpainting...")
        # This function now returns the final 3D numpy array
        final_volume_3d = infill_and_display(model, volume, mask)

        # 4. Post-process (Normalize) the final volume
        print("-" * 20)
        print("Applying normalization to the final volume...")
        try:
            volume_to_save = final_volume_3d.astype(np.float32) # Ensure correct type

            if np.isnan(volume_to_save).any() or np.isinf(volume_to_save).any():
                print("Warning: Final volume contains NaN/Inf. Replacing with 0.")
                volume_to_save = np.nan_to_num(volume_to_save, nan=0.0, posinf=0.0, neginf=0.0)

            p2, p98 = np.percentile(volume_to_save, (2, 98))
            print(f"  Final volume percentiles (P2, P98): ({p2:.4f}, {p98:.4f})")
            volume_to_save = np.clip(volume_to_save, p2, p98)
            range_val = p98 - p2
            if not np.isclose(range_val, 0):
                volume_to_save = (volume_to_save - p2) / range_val # Scale to [0, 1]
                volume_to_save = (volume_to_save * 2.0) - 1.0 # Scale to [-1, 1]
            else:
                volume_to_save.fill(-1.0) # Set to -1 if range is zero
                print("Warning: P2/P98 close. Scaled data might be uniform (-1).")
            print("Normalization complete.")

        except Exception as e:
            print(f"Error during final normalization: {e}. Attempting to save unnormalized data.")
            volume_to_save = final_volume_3d.astype(np.float32) # Fallback

        # 5. Determine SEGY Start Geometry
        # Use min from loaded SEGY geometry if available, otherwise default to 1
        if input_inline_vals is not None and len(input_inline_vals) > 0:
             inline_start_val = max(1, int(np.min(input_inline_vals)))
        else:
             inline_start_val = 1
             print("Warning: Could not determine original inline start, defaulting to 1.")

        if input_crossline_vals is not None and len(input_crossline_vals) > 0:
             crossline_start_val = max(1, int(np.min(input_crossline_vals)))
        else:
             crossline_start_val = 1
             print("Warning: Could not determine original crossline start, defaulting to 1.")

        print(f"Using inline_start={inline_start_val}, crossline_start={crossline_start_val} for SEGY output.")

        # 6. Write Output SEGY File
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        output_segy_path = output_dir / output_segy_filename
        print(f"Writing final SEGY file to: {output_segy_path.resolve()}")
        try:
            # Use the stable revert version of test_segy_io
            test_segy_io.write_3d_numpy_to_segy_with_coordinates(
                filename=str(output_segy_path),
                data=volume_to_save,
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

        print("\nProcessing finished.")
    else:
        print("Failed to load any data. Exiting.")

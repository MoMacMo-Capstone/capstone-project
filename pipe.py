import torch
import torch.nn as nn
import numpy as np
import draw_mask
from model import CombinedGenerator, device, resolution
import os
from pathlib import Path
import test_segy_io

# Select the desired model checkpoint
# G = CombinedGenerator("trained_128x128_b54a594.ckpt").to(device)
# G = CombinedGenerator("trained_64x64_cbbfce7_div1.ckpt").to(device)
G = CombinedGenerator("trained_64x64_305ee13_div1.ckpt", map_location=device).to(device) # Added map_location for robustness

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
    model.eval()
    infilled_volume = volume.copy()
    noise_volume = np.zeros_like(volume)
    mean_volume = volume.copy()
    stdev_volume = np.zeros_like(volume)

    output_slice_dir = Path("./inpainted_slices")
    output_slice_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving inpainted slices to: {output_slice_dir.resolve()}")

    mask_applied = np.any(mask)
    if mask_applied:
        try:
            top, left, bottom, right = center_of_mass_and_rectangle(mask, resolution)
            print(f"Calculated bounding box for inpainting: top={top}, left={left}, bottom={bottom}, right={right}")
            if (bottom - top) != resolution[0] or (right - left) != resolution[1]:
                 print(f"Warning: Extracted patch size ({bottom-top}x{right-left}) doesn't match model resolution ({resolution[0]}x{resolution[1]}). Check mask or centering logic.")

        except ValueError as e:
            print(f"Error calculating bounding box: {e}. Skipping inpainting.")
            mask_applied = False # Treat as if no mask was applied

    with torch.no_grad():
        for i in range(volume.shape[2]):
            slice_input = volume[:, :, i]
            slice_filename = output_slice_dir / f"slice_z_{i}.npy"

            if not mask_applied:
                np.save(slice_filename, slice_input)
                if i == 0: print(f"No mask applied or bounding box error, saving original slices...")
                continue

            # Process the cropped region
            cropped_input = slice_input[top:bottom, left:right]
            # Ensure the cropped mask matches the input size for the model
            cropped_mask = mask[top:bottom, left:right]

            input_tensor = torch.tensor(cropped_input, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            mask_tensor = torch.tensor(cropped_mask, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)

            infill, noise, mean, stdev = model.forward_with_intermediate(input_tensor, mask_tensor)
            infill_np = infill.squeeze().cpu().numpy()
            noise_np = noise.squeeze().cpu().numpy()
            mean_np = mean.squeeze().cpu().numpy()
            stdev_np = stdev.squeeze().cpu().numpy()

            current_infilled_slice = slice_input.copy()
            # Place the inpainted result back into the full slice using the boolean mask
            current_infilled_slice[top:bottom, left:right][cropped_mask] = infill_np[cropped_mask]

            # Store results for the full volume display (optional)
            infilled_volume[:, :, i] = current_infilled_slice
            noise_volume[top:bottom, left:right, i] = noise_np
            mean_volume[top:bottom, left:right, i] = mean_np
            stdev_volume[top:bottom, left:right, i] = stdev_np

            # Save the complete slice (original data + inpainted patch)
            np.save(slice_filename, current_infilled_slice)

            if (i + 1) % 50 == 0 or i == volume.shape[2] - 1: # Print progress less often
                print(f"Infilled and saved slice {i + 1} of {volume.shape[2]}")

    # Optional display
    print("Displaying results (optional)...")
    try:
        draw_mask.show_volume_with_slider(volume, mask, infilled_volume, noise_volume, mean_volume, stdev_volume)
        print("Close the display window to finish.")
    except Exception as e:
        print(f"Could not display results: {e}")


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
            except FileNotFoundError:
                print(f"Error: Default BIN file '{bin_filename}' not found.")
            except Exception as e:
                print(f"Error reading BIN file '{bin_filename}': {e}")

        elif load_choice == '2':
            segy_filename = input("Enter the path to the SEGY file: ")
            data = load_data_from_segy(segy_filename)

        else:
            print("Invalid choice. Please enter 1 or 2.")

    if data is not None:
        print("-" * 20)
        volume = draw_mask.choose_volume(data)
        print(f"Selected volume shape for processing: {volume.shape}")

        print("Please draw the mask for inpainting...")
        mask = draw_mask.make_drawn_mask_2d(volume)
        print("Mask created. Starting inpainting process...")
        infill_and_display(model, volume, mask)
        print("Processing finished.")
    else:
        print("Failed to load any data. Exiting.")

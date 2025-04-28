import numpy as np
import segyio
import os

def write_3d_numpy_to_segy_with_coordinates(
    filename: str,
    data: np.ndarray,
    client: str = "MOMACMO LIMITED",
    sample_interval_us: int = 2000,
    inline_start: int = 1, # Default, expected to be >= 1
    crossline_start: int = 1, # Default, expected to be >= 1
    inline_step: int = 1,
    crossline_step: int = 1,
    origin: tuple = (0.0, 0.0),    # (y0, x0) - Default origin
    spacing: tuple = (25.0, 25.0),  # (dy, dx) - Default spacing
    x_axis_azimuth: float = 90.0,  # Default azimuth
    coord_scalar: int = -100,      # Default scalar (-100 means divide by 100)
    projection: str = "UNKNOWN",
    provider: str = "MOMACMO LIMITED"
):
    """
    Write a 3D numpy array to a SEGY file using segyio.
    (Stable Revert: Includes coordinates and trace sample interval fix)

    Parameters:
        filename (str): Output SEGY file path
        data (np.ndarray): 3D array with shape (n_inline, n_crossline, n_samples)
        client (str): Client for the data (EBCDIC header)
        sample_interval_us (int): Sample interval in microseconds
        inline_start (int): Starting inline number (should be >= 1)
        crossline_start (int): Starting crossline number (should be >= 1)
        inline_step (int): Step between inlines
        crossline_step (int): Step between crosslines
        origin (tuple): (y0, x0) Default origin coordinates for calculating CDP_X/Y
        spacing (tuple): (dy, dx) Default spacing for calculating CDP_X/Y
        x_axis_azimuth (float): Azimuth for EBCDIC header
        coord_scalar (int): SEGY coordinate scalar for CDP_X/Y
        projection (str): Projection info (EBCDIC header)
        provider (str): Provider info (EBCDIC header)
    """
    if data.ndim != 3:
        raise ValueError(f"Input data must be 3D (inline, crossline, samples). Got shape: {data.shape}")
    if sample_interval_us <= 0:
        raise ValueError(f"sample_interval_us must be positive. Got: {sample_interval_us}")
    if inline_start <= 0 or crossline_start <= 0:
         print(f"Warning: inline_start ({inline_start}) or crossline_start ({crossline_start}) is <= 0. SEGY standard usually expects > 0.")

    n_inline, n_crossline, n_samples = data.shape
    n_traces = n_inline * n_crossline
    print(f"Preparing SEGY file '{filename}' with:")
    print(f"  - Shape: (inline={n_inline}, crossline={n_crossline}, samples={n_samples})")
    print(f"  - Total traces: {n_traces}")
    print(f"  - Sample Interval: {sample_interval_us} us")
    print(f"  - Inline Range: {inline_start} to {inline_start + (n_inline-1)*inline_step} (Step: {inline_step})")
    print(f"  - Crossline Range: {crossline_start} to {crossline_start + (n_crossline-1)*crossline_step} (Step: {crossline_step})")

    # Ensure data is C-contiguous float32
    flat_data = np.require(data.reshape(n_traces, n_samples), dtype=np.float32, requirements=['C'])

    # Define SEGY spec
    spec = segyio.spec()
    # Let segyio handle sorting based on loop order (Inline-major)
    spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
    spec.format = segyio.SegySampleFormat.IEEE_FLOAT_4_BYTE # 5 = IEEE float
    spec.samples = np.arange(n_samples) * (sample_interval_us / 1e6) # sample times in seconds
    spec.tracecount = n_traces

    # Unpack origin and spacing for coordinate calculation
    y0, x0 = origin
    dy, dx = spacing

    with segyio.create(filename, spec) as segyfile:
        # --- Populate Binary Header ---
        segyfile.bin[segyio.BinField.Traces] = n_traces
        segyfile.bin[segyio.BinField.Samples] = n_samples
        segyfile.bin[segyio.BinField.Interval] = sample_interval_us
        segyfile.bin[segyio.BinField.Format] = 5  # 5 = IEEE float 4-byte (Implies fixed length)
        # Removed explicit SortingCode = 3 setting
        segyfile.bin[segyio.BinField.SEGYRevision] = 1
        segyfile.bin[segyio.BinField.ExtendedHeaders] = 0

        # --- Populate Trace Headers (With Coordinates) and Write Trace Data ---
        trace_index = 0
        for i in range(n_inline): # Loop through inlines
            for j in range(n_crossline): # Loop through crosslines
                inline = inline_start + i * inline_step
                crossline = crossline_start + j * crossline_step

                # Calculate Map Coordinates
                y = y0 + i * dy
                x = x0 + j * dx

                # Apply coordinate scalar
                if coord_scalar == 0:
                    y_scaled = int(round(y))
                    x_scaled = int(round(x))
                else:
                    scale_factor = abs(coord_scalar)
                    y_scaled = int(round(np.nan_to_num(float(y) * scale_factor)))
                    x_scaled = int(round(np.nan_to_num(float(x) * scale_factor)))

                # Writing Trace Headers
                segyfile.header[trace_index] = {
                    # Essential Geometry
                    segyio.TraceField.INLINE_3D: inline,
                    segyio.TraceField.CROSSLINE_3D: crossline,

                    # Map Coordinates
                    segyio.TraceField.CDP_Y: y_scaled,
                    segyio.TraceField.CDP_X: x_scaled,
                    segyio.TraceField.SourceGroupScalar: coord_scalar,

                    # Essential Data Info - CRITICAL FIX RETAINED
                    segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: sample_interval_us,

                    # Essential Sequence Number
                    segyio.TraceField.TRACE_SEQUENCE_LINE: trace_index + 1,

                    # Optional but standard
                    segyio.TraceField.TraceNumber: trace_index + 1,
                    segyio.TraceField.FieldRecord: 1,
                    segyio.TraceField.CDP: trace_index + 1,

                    # Optional Duplicates
                    segyio.TraceField.SourceY: y_scaled,
                    segyio.TraceField.SourceX: x_scaled,
                    segyio.TraceField.GroupY: y_scaled,
                    segyio.TraceField.GroupX: x_scaled,
                    segyio.TraceField.offset: 0,
                }

                # Write the actual trace sample data
                segyfile.trace[trace_index] = flat_data[trace_index]

                trace_index += 1

        # --- Add EBCDIC Textual Header ---
        ebcdic_lines = [
            f"C 1 CLIENT: {client[:68]}",
            f"C 2 DATA WRITTEN WITH PYTHON SEGYIO",
            f"C 3 INLINE/CROSSLINE ORDERING ASSUMED", # Removed explicit sort code mention
            f"C 4 INLINE_3D START: {inline_start} STEP: {inline_step} COUNT: {n_inline}",
            f"C 5 CROSSLINE_3D START: {crossline_start} STEP: {crossline_step} COUNT: {n_crossline}",
            f"C 6 SAMPLE INTERVAL (US): {sample_interval_us} SAMPLES/TRACE: {n_samples}",
            f"C 7 COORDINATE ORIGIN (Y, X): ({origin[0]:.3f}, {origin[1]:.3f})",
            f"C 8 COORDINATE SPACING (DY, DX): ({spacing[0]:.3f}, {spacing[1]:.3f})",
            f"C 9 COORDINATES SCALED BY: {coord_scalar} (NEGATIVE = DIVIDE)",
            f"C10 CDP_X/CDP_Y = SourceX/SourceY = GroupX/GroupY (ASSUMED ZERO-OFFSET)",
            f"C11 PROJECTION: {projection[:66]}",
            f"C12 X-AXIS AZIMUTH (DEGREES EAST OF NORTH): {x_axis_azimuth:.2f}",
            f"C13 GENERATED BY: {provider[:62]}"
        ]
        for i in range(len(ebcdic_lines) + 1, 41):
            ebcdic_lines.append(f"C{i:<2d}")
        segyfile.text[0] = segyio.tools.create_text_header(ebcdic_lines)

    print(f"SEGY file successfully written to: {os.path.abspath(filename)}")


# --- Keep the read and test functions below ---
# (They should be compatible with this writer version)
def read_segy_to_3d_numpy(filename):
    # This reader should now work again with coordinate fields present
    with segyio.open(filename, "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = f.samples.size

        inlines = f.attributes(segyio.TraceField.INLINE_3D)[:]
        crosslines = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]
        # Try reading coordinates, handle potential missing scalar
        try:
            cdp_y = f.attributes(segyio.TraceField.CDP_Y)[:]
            cdp_x = f.attributes(segyio.TraceField.CDP_X)[:]
            scalar_attr = f.attributes(segyio.TraceField.SourceGroupScalar)
            scalar = scalar_attr[0] if scalar_attr is not None and len(scalar_attr) > 0 else 1
            if scalar == 0: scalar = 1
            scale = 1.0 / abs(scalar) if scalar < 0 else abs(scalar)
            coords_present = True
        except Exception as e:
            print(f"Warning reading coordinate headers: {e}. Coordinates will be None.")
            coords_present = False
            scale = 1.0

        inline_vals = np.unique(inlines)
        crossline_vals = np.unique(crosslines)
        n_inline = inline_vals.size
        n_crossline = crossline_vals.size

        if n_traces != n_inline * n_crossline:
             print(f"Warning: Trace count ({n_traces}) does not match unique inline/crossline grid size ({n_inline} x {n_crossline}).")

        try:
            data = segyio.tools.cube(f)
            if data.shape != (n_inline, n_crossline, n_samples):
                 print(f"Warning: segyio.tools.cube shape {data.shape} differs from expected ({n_inline}, {n_crossline}, {n_samples}).")
        except Exception as e:
             print(f"Warning: segyio.tools.cube failed ({e}). Reading trace by trace.")
             data = np.zeros((n_inline, n_crossline, n_samples), dtype=np.float32)
             inline_map = {v: i for i, v in enumerate(inline_vals)}
             crossline_map = {v: j for j, v in enumerate(crossline_vals)}
             for i in range(n_traces):
                  il = inlines[i]
                  xl = crosslines[i]
                  if il in inline_map and xl in crossline_map:
                       ii = inline_map[il]
                       jj = crossline_map[xl]
                       data[ii, jj, :] = f.trace[i]
                  else:
                       print(f"Warning: Skipping trace {i+1} with inconsistent inline/crossline ({il}/{xl}).")

        coords = np.zeros((n_inline, n_crossline, 2), dtype=np.float64) if coords_present else None
        if coords_present:
            inline_map = {v: i for i, v in enumerate(inline_vals)}
            crossline_map = {v: j for j, v in enumerate(crossline_vals)}
            for i in range(n_traces):
                il = inlines[i]
                xl = crosslines[i]
                if il in inline_map and xl in crossline_map:
                    ii = inline_map[il]
                    jj = crossline_map[xl]
                    coords[ii, jj, 0] = cdp_y[i] * scale
                    coords[ii, jj, 1] = cdp_x[i] * scale

        return data, inline_vals, crossline_vals, coords


def test_segy_round_trip():
    print("\n--- Running SEGY Round Trip Test (Stable Revert) ---")
    np.random.seed(42)
    test_cube = (np.random.rand(5, 6, 10) * 100).astype(np.float32)

    test_dir = Path("./reconstructed_test")
    test_dir.mkdir(exist_ok=True)
    filename = str(test_dir / "test_stable_revert_output.sgy")

    inline_start_test=100
    crossline_start_test=200
    origin_test = (1000.0, 5000.0)
    spacing_test = (10.0, 12.5)
    coord_scalar_test = -10

    write_3d_numpy_to_segy_with_coordinates(
        filename,
        test_cube,
        inline_start=inline_start_test,
        crossline_start=crossline_start_test,
        origin=origin_test,
        spacing=spacing_test,
        coord_scalar=coord_scalar_test,
    )

    read_cube, inlines, crosslines, coords = read_segy_to_3d_numpy(filename)

    assert read_cube.shape == test_cube.shape, f"Shape mismatch: Got {read_cube.shape}, expected {test_cube.shape}"
    assert np.allclose(read_cube, test_cube), "Data mismatch"

    if inlines is not None:
         assert np.all(inlines == np.arange(inline_start_test, inline_start_test + test_cube.shape[0])), "Inline index mismatch"
    if crosslines is not None:
         assert np.all(crosslines == np.arange(crossline_start_test, crossline_start_test + test_cube.shape[1])), "Crossline index mismatch"

    if coords is not None:
         print("Checking coordinates...")
         scale_test = 1.0 / abs(coord_scalar_test) if coord_scalar_test < 0 else abs(coord_scalar_test)
         for i, il in enumerate(inlines):
              for j, xl in enumerate(crosslines):
                   expected_y = origin_test[0] + i * spacing_test[0]
                   expected_x = origin_test[1] + j * spacing_test[1]
                   actual_y, actual_x = coords[i, j]
                   assert np.isclose(expected_y, actual_y), f"Y coordinate mismatch at ({i}, {j})"
                   assert np.isclose(expected_x, actual_x), f"X coordinate mismatch at ({i}, {j})"
         print("Coordinate check passed.")
    else:
         print("Coordinates were not read, skipping check.")

    print("--- SEGY round-trip test (Stable Revert) passed! ---")


if __name__ == "__main__":
    # test_segy_round_trip()
    pass
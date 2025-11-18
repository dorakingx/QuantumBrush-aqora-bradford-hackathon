import numpy as np
import colorsys
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, partial_trace, entropy
import importlib.util

spec = importlib.util.spec_from_file_location("utils", "effect/utils.py")
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def create_bell_state():
    """
    Create a Bell state (maximally entangled state) |Φ+⟩ = (|00⟩ + |11⟩)/√2
    """
    qc = QuantumCircuit(2)
    qc.h(0)  # Apply Hadamard gate to create superposition
    qc.cx(0, 1)  # Apply CNOT to create entanglement
    return qc


def entangle_colors(initial_colors, entanglement_strength=1.0):
    """
    Create quantum entanglement between color pairs and evolve the colors.
    
    Args:
        initial_colors: List of (phi, theta) tuples representing initial color angles
        entanglement_strength: Strength of entanglement effect (0.0 to 1.0)
    
    Returns:
        List of (phi, theta) tuples representing evolved color angles
    """
    n_colors = len(initial_colors)
    if n_colors < 2:
        return initial_colors
    
    # Create pairs of colors to entangle
    circuits = []
    observables_list = []
    
    for i in range(0, n_colors - 1, 2):
        # Get two colors to entangle
        phi1, theta1 = initial_colors[i]
        phi2, theta2 = initial_colors[i + 1]
        
        # Create quantum circuit with 2 qubits (one for each color)
        qc = QuantumCircuit(2)
        
        # Prepare initial states for both colors
        # Color 1: qubit 0
        qc.ry(theta1, 0)
        qc.rz(phi1, 0)
        
        # Color 2: qubit 1
        qc.ry(theta2, 1)
        qc.rz(phi2, 1)
        
        # Create entanglement between the two colors using Bell state
        qc.h(0)
        qc.cx(0, 1)
        
        # Apply controlled rotations based on entanglement strength
        if entanglement_strength > 0:
            # Controlled phase rotation to mix the colors
            qc.crz(entanglement_strength * np.pi / 2, 0, 1)
        
        circuits.append(qc)
        
        # Measure X, Y, Z for each qubit
        ops = []
        for qubit in range(2):
            for pauli in ['X', 'Y', 'Z']:
                pauli_str = 'I' * qubit + pauli + 'I' * (2 - qubit - 1)
                ops.append(SparsePauliOp(pauli_str))
        observables_list.append(ops)
    
    # Run quantum simulation
    all_results = []
    for circuit, observables in zip(circuits, observables_list):
        try:
            results = utils.run_estimator(circuit, observables, backend=None)
            # Extract expectation values from results (each result is a list)
            if isinstance(results, list):
                results = np.array([val[0] if isinstance(val, (list, np.ndarray)) and len(val) > 0 else val for val in results])
            else:
                results = np.array(results)
            all_results.append(results)
        except Exception as e:
            print(f"Error in quantum simulation: {e}")
            # Return original colors if simulation fails
            return initial_colors
    
    # Process results and extract new angles
    final_colors = []
    result_idx = 0
    
    for i in range(0, n_colors - 1, 2):
        if i + 1 < n_colors and result_idx < len(all_results):
            results = all_results[result_idx]
            result_idx += 1
            
            # Extract expectations for qubits 0 and 1 (entangled pair)
            # Each qubit has X, Y, Z observables in order
            x1 = results[0]  # X for qubit 0
            y1 = results[1]  # Y for qubit 0
            z1 = results[2]  # Z for qubit 0
            
            x2 = results[3]   # X for qubit 1
            y2 = results[4]   # Y for qubit 1
            z2 = results[5]   # Z for qubit 1
            
            # Convert to spherical coordinates
            phi1_new = np.arctan2(y1, x1) % (2 * np.pi)
            theta1_new = np.arctan2(np.sqrt(x1**2 + y1**2), z1) % np.pi
            
            phi2_new = np.arctan2(y2, x2) % (2 * np.pi)
            theta2_new = np.arctan2(np.sqrt(x2**2 + y2**2), z2) % np.pi
            
            final_colors.append((phi1_new, theta1_new))
            final_colors.append((phi2_new, theta2_new))
        else:
            # If odd number of colors, keep the last one unchanged
            final_colors.append(initial_colors[i])
    
    # Handle case where we have an odd number of colors
    if n_colors % 2 == 1:
        final_colors.append(initial_colors[-1])
    
    return final_colors


def run(params):
    """
    Executes the EntangleBrush quantum effect pipeline.
    
    This brush uses quantum entanglement to create correlated color changes
    across different parts of the stroke. Colors are paired and entangled,
    so that changes in one color affect its entangled partner.
    
    Args:
        params (dict): A dictionary containing all the relevant data.
            - stroke_input: Contains image_rgba and path
            - user_input: Contains Radius, Strength, Entanglement Pairs, and Color
    
    Returns:
        np.ndarray: The new numpy array of RGBA values
    """
    
    # Extract image to work from
    image = params["stroke_input"]["image_rgba"].copy()
    assert image.shape[-1] == 4, "Image must be RGBA format"
    
    height = image.shape[0]
    width = image.shape[1]
    
    path = params["stroke_input"]["path"]
    
    # Get user parameters
    radius = params["user_input"]["Radius"]
    assert radius > 0, "Radius must be greater than 0"
    
    strength = params["user_input"]["Strength"]
    assert 0.0 <= strength <= 1.0, "Strength must be between 0.0 and 1.0"
    
    entanglement_pairs = params["user_input"]["Entanglement Pairs"]
    assert entanglement_pairs > 0, "Entanglement Pairs must be greater than 0"
    
    base_color = params["user_input"]["Color"]
    base_phi, base_theta, base_s = utils.color_to_spherical(base_color)
    
    # Split path into segments based on number of entanglement pairs
    path_length = len(path)
    n_segments = entanglement_pairs * 2  # Each pair has 2 segments
    
    if path_length < n_segments:
        n_segments = max(2, path_length // 2)
    
    split_size = max(1, path_length // n_segments)
    split_paths = []
    for i in range(n_segments - 1):
        split_paths.append(path[i * split_size : (i + 1) * split_size])
    split_paths.append(path[(n_segments - 1) * split_size :])
    
    # Extract initial colors from each segment
    initial_angles = []
    segment_regions = []
    
    for segment_path in split_paths:
        region = utils.points_within_radius(segment_path, radius, border=(height, width))
        segment_regions.append(region)
        
        # Get average color from region
        if len(region) > 0:
            region_colors = image[region[:, 0], region[:, 1]].astype(np.float32) / 255.0
            region_hls = utils.rgb_to_hls(region_colors)
            
            # Calculate mean hue, lightness, saturation
            mean_h = np.mean(region_hls[..., 0]) % 1.0
            mean_l = np.mean(region_hls[..., 1])
            mean_s = np.mean(region_hls[..., 2])
            
            # Convert to spherical coordinates
            phi = 2 * np.pi * mean_h
            theta = np.pi * mean_l
            
            initial_angles.append((phi, theta))
        else:
            # Use base color if region is empty
            initial_angles.append((base_phi, base_theta))
    
    # Apply quantum entanglement
    entanglement_strength = strength
    final_angles = entangle_colors(initial_angles, entanglement_strength)
    
    # Apply the evolved colors back to the image
    for i, (region, (phi_new, theta_new)) in enumerate(zip(segment_regions, final_angles)):
        if len(region) == 0:
            continue
        
        # Get original colors
        original_colors = image[region[:, 0], region[:, 1]].astype(np.float32) / 255.0
        original_hls = utils.rgb_to_hls(original_colors)
        
        # Get initial angle for this segment
        phi_old, theta_old = initial_angles[i]
        
        # Calculate color shifts
        h_shift = (phi_new - phi_old) / (2 * np.pi)
        l_shift = (theta_new - theta_old) / np.pi
        
        # Apply shifts with strength control
        new_hls = original_hls.copy()
        new_hls[..., 0] = (new_hls[..., 0] + strength * h_shift) % 1.0
        new_hls[..., 1] = np.clip(new_hls[..., 1] + strength * l_shift, 0.0, 1.0)
        
        # Convert back to RGB
        new_rgb = utils.hls_to_rgb(new_hls)
        new_rgb = (new_rgb * 255).astype(np.uint8)
        
        # Apply to image
        image[region[:, 0], region[:, 1]] = new_rgb
    
    print("EntangleBrush effect applied successfully")
    return image


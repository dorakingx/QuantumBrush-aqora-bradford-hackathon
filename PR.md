# EntangleBrush: Quantum Entanglement-Based Brush for QuantumBrush

## Overview

This pull request introduces **EntangleBrush**, a new quantum algorithm-based brush that leverages quantum entanglement to create correlated color transformations across different parts of a stroke. This brush demonstrates the non-local correlations inherent in quantum entanglement, where measurements on one part of an entangled system instantaneously affect the other.

## Repository

[QuantumBrush-aqora-bradford-hackathon](https://github.com/dorakingx/QuantumBrush-aqora-bradford-hackathon.git)

## Features

### Quantum Entanglement Algorithm

EntangleBrush uses Bell states and CNOT gates to create quantum entanglement between color pairs:

- **Bell State Creation**: Each pair of stroke segments is entangled using the maximally entangled Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
- **Color Correlation**: Colors in entangled pairs influence each other, creating unique visual effects where color changes in one segment cause corresponding changes in its entangled partner
- **Quantum Evolution**: The entanglement strength parameter controls how much the quantum evolution affects the original colors

### Technical Implementation

- **Quantum Circuit**: 2-qubit circuits for each color pair (simplified from initial 4-qubit design for efficiency)
- **Observables**: X, Y, Z Pauli operators measured for each qubit to extract color information
- **State Preparation**: Initial colors are converted to quantum states using spherical coordinates (phi, theta)
- **Error Handling**: Robust error handling ensures graceful fallback if quantum simulation fails

### Parameters

- **Radius** (1-100): Determines the size of the brush effect
- **Strength** (0.0-1.0): Controls how much quantum evolution affects colors
  - 0.0 = subtle quantum-inspired variations
  - 1.0 = dramatic quantum behavior takes over
- **Entanglement Pairs** (1-20): Number of color pairs to create and entangle
- **Color**: Base color for initialization

## Files Added

```
effect/entanglebrush/
├── __init__.py
├── entanglebrush.py
└── entanglebrush_requirements.json
```

## Code Structure

### Main Functions

1. **`entangle_colors(initial_colors, entanglement_strength)`**
   - Creates quantum entanglement between color pairs
   - Runs quantum simulation using Qiskit
   - Returns evolved color angles

2. **`run(params)`**
   - Main entry point for the brush effect
   - Processes stroke path and segments
   - Applies quantum-entangled color transformations

### Dependencies

- `numpy >= 2.1.0`
- `qiskit >= 1.0.0`
- `qiskit_ibm_runtime >= 0.20.0`
- `qiskit_aer >= 0.17.0`
- `colorsys` (standard library)

## Usage

1. Select EntangleBrush from the Control Panel
2. Adjust parameters:
   - Set **Radius** for brush size
   - Set **Strength** for quantum effect intensity
   - Set **Entanglement Pairs** for number of correlated segments
   - Choose **Color** for base initialization
3. Click and drag on the canvas to create a stroke
4. The quantum entanglement creates non-local correlations between paired segments

## Examples

### Ukiyo-e Artwork Transformation

The following example demonstrates EntangleBrush applied to a traditional Japanese ukiyo-e artwork:

**Original Image:**
![Original Ukiyo-e](project/project_1763508961929/original.png)

**After Applying EntangleBrush:**
![Quantum-Enhanced Ukiyo-e](project/project_1763508961929/current.png)

The quantum entanglement effect creates unique color correlations across different parts of the artwork, demonstrating how quantum mechanics can be used to create artistic transformations while preserving the essence of the original piece.

## Scientific Background

This brush demonstrates quantum entanglement, a fundamental phenomenon in quantum mechanics where:

- Two quantum systems become correlated in such a way that the state of one cannot be described independently of the other
- Measurements on one part of an entangled system instantaneously affect the other, even when spatially separated
- This "spooky action at a distance" (as Einstein called it) is a real physical phenomenon that has been experimentally verified

The brush visualizes this by:
1. Converting color information to quantum states
2. Creating entangled pairs using Bell states
3. Evolving the quantum states
4. Converting the evolved states back to colors

## Testing

The brush has been tested with:
- Python 3.11+
- Qiskit 2.2.3
- NumPy 2.3.5
- macOS Sequoia (15.5)

## Future Enhancements

Potential improvements:
- Support for multi-qubit entanglement (GHZ states, W states)
- Adaptive entanglement based on stroke geometry
- Real-time quantum hardware execution
- Visualization of entanglement entropy

## Author

QuantumBrush Developer

## License

Apache License 2.0 (same as QuantumBrush project)


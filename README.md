# Quanvolutional Image Filter Model 🚀

A hybrid quantum-classical convolutional neural network (QCNN) implementation for MNIST image classification, featuring multiple quantum filter architectures and a stunning interactive dashboard.

Based on the research paper: *Henderson et al., "Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits" (arXiv:1904.04767)*.

---

## 🌟 Key Features

- **Multi-Filter Bank**: Implements three distinct quantum circuit architectures as convolutional kernels:
  - **Random**: Non-trainable random unitary circuit (surprisingly effective baseline).
  - **Strongly Entangling**: Robust entanglement layers for complex feature mapping.
  - **Custom Entangled**: Hand-designed circuit with ring-CNOTs and CZ cross-connections.
- **Hybrid Pipeline**: Seamlessly integrates **PennyLane** for quantum simulation with **PyTorch** for classical neural network training.
- **Quantum Pre-processing**: Optimized engine that extracts features through quantum kernels and caches them for fast training.
- **Interactive Dashboard**: A glassmorphism HTML dashboard featuring:
  - Dynamic training curves (Chart.js).
  - Detailed quantum circuit diagrams.
  - Filter gallery showing original vs. quanvolved feature maps.
  - Average response heatmaps.
- **Unicode Resilience**: Fully compatible with Windows console environments.

---

## 📊 Experimental Results

| Model | Best Validation Accuracy | Final Loss |
| :--- | :--- | :--- |
| **Quanv [random]** | **96.50%** | **0.2061** |
| Classical CNN | 95.00% | 0.2462 |
| Quanv [strongly_entangling] | 93.00% | 0.3466 |
| Quanv [custom_entangled] | 89.50% | 0.6166 |

> [!NOTE]
> Training performed on a subset of 500 images (30 epochs). The **Random Quantum Filter** demonstrated exceptional performance, outperforming the pure classical CNN baseline.

---

## 🛠️ Installation

Ensure you have Python 3.8+ installed.

```bash
pip install pennylane torch torchvision matplotlib
```

---

## 🚀 Usage

### 1. Run the Model
Execute the main script to start quantum pre-processing, training, and visualization generation:

```bash
# On Windows (ensures UTF-8 encoding for circuit diagrams in console)
$env:PYTHONIOENCODING="utf-8"; python quanvolution_model.py
```

### 2. View the Dashboard
The script generates a `quanv_output` directory. You can view the results in the interactive dashboard:

```bash
# Start a local server
python -m http.server 8080 --directory quanv_output
```
Open your browser and navigate to: [http://localhost:8080/dashboard.html](http://localhost:8080/dashboard.html)

---

## 📁 Project Structure

```text
QNN3/
├── quanvolution_model.py    # Main source code (Model, Engine, Viz)
├── README.md                # Project documentation
└── quanv_output/            # Generated artifacts
    ├── dashboard.html       # Visual dashboard
    ├── metrics.json         # Numerical results
    ├── plots/               # PNG visualizations
    └── q_train_*.npy        # Cached quantum feature maps
```

---

## 🧬 Architecture Overview

1. **Encoding**: Scales 2×2 pixel patches into rotation angles ($\pi \phi$).
2. **Quantum Filter**: Applies parameterized rotations and entangling gates.
3. **Measurement**: Extracts 4 PauliZ expectation values per patch.
4. **Reshape**: Produces a 14×14×4 multi-channel feature map.
5. **Classification**: Feeds features into a classical CNN head for digit recognition.

---

## 🎓 References

- [PennyLane Documentation](https://pennylane.ai/)
- [Quanvolutional Neural Networks (Henderson et al.)](https://arxiv.org/abs/1904.04767)

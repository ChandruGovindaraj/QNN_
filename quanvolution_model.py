"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               QUANVOLUTIONAL IMAGE FILTER MODEL                            ║
║   Hybrid Quantum-Classical Convolutional Neural Network for MNIST          ║
║                                                                            ║
║   Based on: Henderson et al., "Quanvolutional Neural Networks:             ║
║   Powering Image Recognition with Quantum Circuits" (arXiv:1904.04767)     ║
║                                                                            ║
║   Implementation: PennyLane + PyTorch (modern, non-deprecated stack)       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import numpy as np
import pennylane as qml
from pennylane.templates import RandomLayers, StronglyEntanglingLayers

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "n_qubits": 4,                  # Qubits per quanvolution kernel (2x2 patch)
    "n_layers": 1,                  # Depth of random/variational layers
    "kernel_size": 2,               # Patch size (kernel_size x kernel_size)
    "stride": 2,                    # Stride of the quanvolution
    "n_epochs": 30,                 # Training epochs
    "batch_size": 16,               # Mini-batch size
    "learning_rate": 0.005,         # Adam learning rate
    "n_train": 500,                 # Number of training samples
    "n_test": 200,                  # Number of test samples
    "seed": 42,                     # Reproducibility seed
    "save_dir": "quanv_output",     # Output directory
    "preprocess": True,             # Run quantum pre-processing (or load cached)
    "filter_types": [               # Quantum filter architectures to compare
        "random",                   #   Random unitary (non-trainable)
        "strongly_entangling",      #   Strongly entangling layers
        "custom_entangled",         #   Custom hand-designed entangled circuit
    ],
}


def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Quantum Filter Circuits
# ─────────────────────────────────────────────────────────────────────────────

class QuantumFilterBank:
    """
    A bank of different quantum circuits that each act as a convolutional
    kernel.  Each circuit maps a 2×2 pixel patch (4 values) → 4 expectation
    values (one per qubit), producing 4 output channels.
    """

    def __init__(self, n_qubits=4, n_layers=1, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.seed = seed
        np.random.seed(seed)

        # Pre-generate random parameters for each filter type
        self.rand_params = np.random.uniform(
            high=2 * np.pi, size=(n_layers, n_qubits)
        )
        self.se_params = np.random.uniform(
            high=2 * np.pi, size=(n_layers, n_qubits, 3)
        )
        self.custom_params = np.random.uniform(
            high=2 * np.pi, size=(n_layers, n_qubits * 3)
        )

        # Build PennyLane devices and QNodes
        self.filters = {}
        self._build_filters()

    def _build_filters(self):
        """Create a QNode for each filter type."""

        # ── 1) Random Layers Filter ──────────────────────────────────
        dev_rand = qml.device("default.qubit", wires=self.n_qubits)
        rand_params = self.rand_params.copy()

        @qml.qnode(dev_rand, interface="numpy")
        def random_filter(phi):
            for j in range(self.n_qubits):
                qml.RY(np.pi * phi[j], wires=j)
            RandomLayers(rand_params, wires=list(range(self.n_qubits)))
            return [qml.expval(qml.PauliZ(j)) for j in range(self.n_qubits)]

        self.filters["random"] = random_filter

        # ── 2) Strongly Entangling Layers Filter ─────────────────────
        dev_se = qml.device("default.qubit", wires=self.n_qubits)
        se_params = self.se_params.copy()

        @qml.qnode(dev_se, interface="numpy")
        def strongly_entangling_filter(phi):
            for j in range(self.n_qubits):
                qml.RY(np.pi * phi[j], wires=j)
            StronglyEntanglingLayers(se_params, wires=list(range(self.n_qubits)))
            return [qml.expval(qml.PauliZ(j)) for j in range(self.n_qubits)]

        self.filters["strongly_entangling"] = strongly_entangling_filter

        # ── 3) Custom Entangled Filter ───────────────────────────────
        dev_custom = qml.device("default.qubit", wires=self.n_qubits)
        custom_params = self.custom_params.copy()

        @qml.qnode(dev_custom, interface="numpy")
        def custom_entangled_filter(phi):
            # Angle embedding
            for j in range(self.n_qubits):
                qml.RY(np.pi * phi[j], wires=j)
            # Custom entangling blocks
            for layer in range(self.n_layers):
                base = layer * self.n_qubits * 3
                for j in range(self.n_qubits):
                    qml.RZ(custom_params[layer, j * 3 + 0], wires=j)
                    qml.RX(custom_params[layer, j * 3 + 1], wires=j)
                    qml.RZ(custom_params[layer, j * 3 + 2], wires=j)
                # Ring of CNOTs
                for j in range(self.n_qubits):
                    qml.CNOT(wires=[j, (j + 1) % self.n_qubits])
                # CZ cross-connections
                qml.CZ(wires=[0, 2])
                qml.CZ(wires=[1, 3])
            return [qml.expval(qml.PauliZ(j)) for j in range(self.n_qubits)]

        self.filters["custom_entangled"] = custom_entangled_filter

    def get_filter(self, name):
        """Return a specific quantum filter by name."""
        if name not in self.filters:
            raise ValueError(f"Unknown filter: {name}. Available: {list(self.filters.keys())}")
        return self.filters[name]

    def get_circuit_diagram(self, name):
        """Draw the circuit for a given filter."""
        filt = self.get_filter(name)
        dummy_input = np.array([0.5, 0.3, 0.7, 0.1])
        return qml.draw(filt)(dummy_input)


# ─────────────────────────────────────────────────────────────────────────────
# Quanvolution Engine
# ─────────────────────────────────────────────────────────────────────────────

class QuanvolutionEngine:
    """
    Applies a quantum circuit as a sliding convolutional kernel over images.
    Each 2×2 patch → 4 expectation values → 4 output channels.
    """

    def __init__(self, circuit_fn, kernel_size=2, stride=2, n_output_channels=4):
        self.circuit_fn = circuit_fn
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_channels = n_output_channels

    def process_image(self, image):
        """
        Apply quanvolution to a single grayscale image.

        Args:
            image: 2D numpy array (H, W) with values in [0, 1].

        Returns:
            3D numpy array (H', W', n_channels).
        """
        h, w = image.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1
        output = np.zeros((out_h, out_w, self.n_channels))

        for i in range(out_h):
            for j in range(out_w):
                row = i * self.stride
                col = j * self.stride
                # Extract the 2×2 patch and flatten
                patch = image[
                    row : row + self.kernel_size,
                    col : col + self.kernel_size,
                ].flatten()

                # Run through quantum circuit
                results = self.circuit_fn(patch)
                for c in range(self.n_channels):
                    output[i, j, c] = results[c]

        return output

    def process_dataset(self, images, label=""):
        """
        Apply quanvolution to an entire dataset of images.

        Args:
            images: numpy array (N, H, W)
            label:  string for progress display

        Returns:
            numpy array (N, H', W', n_channels)
        """
        processed = []
        total = len(images)
        start_time = time.time()

        for idx, img in enumerate(images):
            processed.append(self.process_image(img))
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - idx - 1) / rate if rate > 0 else 0
            print(
                f"\r  {label} [{idx+1}/{total}] "
                f"({rate:.1f} img/s, ~{remaining:.0f}s remaining)   ",
                end="", flush=True
            )

        print()
        return np.array(processed)


# ─────────────────────────────────────────────────────────────────────────────
# Classical & Hybrid PyTorch Models
# ─────────────────────────────────────────────────────────────────────────────

class ClassicalCNN(nn.Module):
    """A small classical CNN baseline for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class QuanvClassifier(nn.Module):
    """
    Classifier that operates on quanvolution-preprocessed images.
    Input shape: (batch, 4, 14, 14)  — 4 quanv channels, 14×14 resolution.
    """

    def __init__(self, n_channels=4, img_size=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * (img_size // 2) * (img_size // 2), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


class SimpleQuanvClassifier(nn.Module):
    """
    Minimal classifier — flatten + single dense layer.
    Matches the PennyLane tutorial approach for fair comparison.
    """

    def __init__(self, n_channels=4, img_size=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * img_size * img_size, 10),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Training Engine
# ─────────────────────────────────────────────────────────────────────────────

class TrainingEngine:
    """Handles training, evaluation, and metrics logging."""

    def __init__(self, model, device="cpu", lr=0.005):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in loader:
            X, y = X.to(self.device), y.to(self.device)
            logits = self.model(X)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, n_epochs, name="Model"):
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")

        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{n_epochs} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
                )

        print(f"  [OK] Final val accuracy: {self.history['val_acc'][-1]:.4f}")
        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Generator
# ─────────────────────────────────────────────────────────────────────────────

class VisualizationGenerator:
    """Creates all plots and the interactive HTML dashboard."""

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

    def plot_quantum_filters(self, original_images, filtered_sets, filter_names):
        """
        Visualize original images and their quanvolution outputs for each
        filter type.
        """
        n_samples = min(4, len(original_images))
        n_channels = 4

        for f_idx, (filtered_imgs, f_name) in enumerate(zip(filtered_sets, filter_names)):
            fig, axes = plt.subplots(
                1 + n_channels, n_samples,
                figsize=(3 * n_samples, 3 * (1 + n_channels))
            )
            fig.suptitle(f"Quanvolutional Filter: {f_name}", fontsize=14, fontweight="bold")

            for k in range(n_samples):
                axes[0, k].imshow(original_images[k], cmap="gray")
                axes[0, k].set_title(f"Input {k+1}", fontsize=10)
                axes[0, k].axis("off")
                if k == 0:
                    axes[0, k].set_ylabel("Original", fontsize=10)

                for c in range(n_channels):
                    axes[c + 1, k].imshow(filtered_imgs[k, :, :, c], cmap="viridis")
                    axes[c + 1, k].axis("off")
                    if k == 0:
                        axes[c + 1, k].set_ylabel(f"Ch {c}", fontsize=10)

            plt.tight_layout()
            path = os.path.join(self.save_dir, "plots", f"filters_{f_name}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  [OK] Saved filter visualization: {path}")

    def plot_training_comparison(self, histories, names):
        """Plot training curves for all models."""
        colors = ["#6366f1", "#ec4899", "#10b981", "#f59e0b", "#ef4444"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, (hist, name) in enumerate(zip(histories, names)):
            c = colors[idx % len(colors)]
            axes[0].plot(hist["val_acc"], color=c, linewidth=2, label=name)
            axes[1].plot(hist["val_loss"], color=c, linewidth=2, label=name)

        axes[0].set_title("Validation Accuracy", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])

        axes[1].set_title("Validation Loss", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(self.save_dir, "plots", "training_comparison.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Saved training comparison: {path}")

    def plot_filter_response_heatmap(self, filtered_sets, filter_names):
        """Create heatmaps showing average filter responses."""
        fig, axes = plt.subplots(len(filter_names), 4, figsize=(14, 3 * len(filter_names)))
        if len(filter_names) == 1:
            axes = axes[np.newaxis, :]

        fig.suptitle("Average Filter Response Heatmaps", fontsize=14, fontweight="bold")

        for f_idx, (filt_imgs, f_name) in enumerate(zip(filtered_sets, filter_names)):
            for c in range(4):
                avg_response = np.mean(filt_imgs[:, :, :, c], axis=0)
                im = axes[f_idx, c].imshow(avg_response, cmap="inferno", aspect="auto")
                axes[f_idx, c].set_title(f"{f_name}\nChannel {c}", fontsize=9)
                axes[f_idx, c].axis("off")
                plt.colorbar(im, ax=axes[f_idx, c], fraction=0.046, pad=0.04)

        plt.tight_layout()
        path = os.path.join(self.save_dir, "plots", "filter_heatmaps.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [OK] Saved filter heatmaps: {path}")

    def generate_html_dashboard(self, histories, names, filter_names, circuit_diagrams, config):
        """Generate a stunning interactive HTML dashboard."""
        # Prepare data for charts
        chart_data = {}
        for hist, name in zip(histories, names):
            chart_data[name] = hist

        # Read plot images as base64
        import base64
        plot_images = {}
        plots_dir = os.path.join(self.save_dir, "plots")
        for fname in os.listdir(plots_dir):
            if fname.endswith(".png"):
                with open(os.path.join(plots_dir, fname), "rb") as f:
                    plot_images[fname] = base64.b64encode(f.read()).decode("utf-8")

        # Build model comparison table rows
        table_rows = ""
        for name, hist in zip(names, histories):
            best_acc = max(hist["val_acc"])
            final_loss = hist["val_loss"][-1]
            best_epoch = hist["val_acc"].index(best_acc) + 1
            is_quantum = "quantum" in name.lower() or "quanv" in name.lower()
            badge = '<span class="badge quantum">Quantum</span>' if is_quantum else '<span class="badge classical">Classical</span>'
            table_rows += f"""
                <tr>
                    <td>{badge} {name}</td>
                    <td>{best_acc:.4f}</td>
                    <td>{final_loss:.4f}</td>
                    <td>{best_epoch}</td>
                </tr>"""

        # Build circuit diagram sections
        circuit_html = ""
        for f_name in filter_names:
            diagram = circuit_diagrams.get(f_name, "N/A")
            circuit_html += f"""
                <div class="circuit-card">
                    <h3>{f_name.replace('_', ' ').title()} Filter</h3>
                    <pre class="circuit-diagram">{diagram}</pre>
                </div>"""

        # Build filter gallery
        filter_gallery = ""
        for f_name in filter_names:
            key = f"filters_{f_name}.png"
            if key in plot_images:
                filter_gallery += f"""
                <div class="filter-section">
                    <h3>{f_name.replace('_', ' ').title()} Filter Output</h3>
                    <img src="data:image/png;base64,{plot_images[key]}" alt="{f_name} filter" />
                </div>"""

        # Heatmap
        heatmap_img = ""
        if "filter_heatmaps.png" in plot_images:
            heatmap_img = f'<img src="data:image/png;base64,{plot_images["filter_heatmaps.png"]}" alt="heatmaps" />'

        # Training comparison
        training_img = ""
        if "training_comparison.png" in plot_images:
            training_img = f'<img src="data:image/png;base64,{plot_images["training_comparison.png"]}" alt="training comparison" />'

        # Prepare chart.js data
        epochs_list = list(range(1, config["n_epochs"] + 1))
        chart_colors = ["#6366f1", "#ec4899", "#10b981", "#f59e0b", "#ef4444"]
        acc_datasets = ""
        loss_datasets = ""
        for idx, (name, hist) in enumerate(zip(names, histories)):
            c = chart_colors[idx % len(chart_colors)]
            acc_datasets += f"""{{
                label: '{name}',
                data: {json.dumps([round(v, 4) for v in hist['val_acc']])},
                borderColor: '{c}',
                backgroundColor: '{c}22',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                pointRadius: 1,
            }},"""
            loss_datasets += f"""{{
                label: '{name}',
                data: {json.dumps([round(v, 4) for v in hist['val_loss']])},
                borderColor: '{c}',
                backgroundColor: '{c}22',
                borderWidth: 2,
                tension: 0.3,
                fill: true,
                pointRadius: 1,
            }},"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Quanvolutional Image Filter — Dashboard</title>
    <meta name="description" content="Interactive dashboard for Quanvolutional Neural Network image filtering with quantum circuit visualizations and training metrics." />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        :root {{
            --bg-primary: #0a0a1a;
            --bg-secondary: #111128;
            --bg-card: rgba(20, 20, 50, 0.7);
            --bg-glass: rgba(255, 255, 255, 0.04);
            --border: rgba(255, 255, 255, 0.08);
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent-purple: #8b5cf6;
            --accent-blue: #6366f1;
            --accent-pink: #ec4899;
            --accent-cyan: #22d3ee;
            --accent-green: #10b981;
            --accent-amber: #f59e0b;
            --gradient-primary: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
            --gradient-card: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05));
            --shadow-glow: 0 0 40px rgba(99, 102, 241, 0.15);
            --radius: 16px;
            --radius-sm: 10px;
        }}

        body {{
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }}

        /* Animated background */
        body::before {{
            content: '';
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(ellipse at 20% 20%, rgba(99, 102, 241, 0.12) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(236, 72, 153, 0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(34, 211, 238, 0.05) 0%, transparent 60%);
            z-index: -1;
            animation: bgPulse 8s ease-in-out infinite alternate;
        }}

        @keyframes bgPulse {{
            0% {{ opacity: 0.7; }}
            100% {{ opacity: 1; }}
        }}

        /* ── Header ────────────────────────────────────────────── */
        .header {{
            text-align: center;
            padding: 60px 24px 40px;
            position: relative;
        }}

        .header::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 200px;
            height: 2px;
            background: var(--gradient-primary);
            border-radius: 2px;
        }}

        .header h1 {{
            font-size: clamp(2rem, 5vw, 3.5rem);
            font-weight: 800;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            letter-spacing: -0.02em;
            margin-bottom: 12px;
        }}

        .header p {{
            color: var(--text-secondary);
            font-size: 1.1rem;
            max-width: 600px;
            margin: 0 auto;
            line-height: 1.6;
        }}

        .header .tag {{
            display: inline-block;
            margin-top: 16px;
            padding: 6px 16px;
            background: var(--bg-glass);
            border: 1px solid var(--border);
            border-radius: 999px;
            font-size: 0.82rem;
            color: var(--accent-cyan);
            font-family: 'JetBrains Mono', monospace;
        }}

        /* ── Navigation ───────────────────────────────────────── */
        .nav {{
            display: flex;
            justify-content: center;
            gap: 8px;
            padding: 20px 24px;
            flex-wrap: wrap;
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(10, 10, 26, 0.85);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border);
        }}

        .nav button {{
            padding: 10px 22px;
            border: 1px solid var(--border);
            border-radius: 999px;
            background: var(--bg-glass);
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            font-size: 0.88rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        .nav button:hover {{
            color: var(--text-primary);
            border-color: var(--accent-purple);
            background: rgba(139, 92, 246, 0.1);
        }}

        .nav button.active {{
            background: var(--gradient-primary);
            color: white;
            border-color: transparent;
            box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
        }}

        /* ── Layout ───────────────────────────────────────────── */
        .container {{
            max-width: 1280px;
            margin: 0 auto;
            padding: 32px 24px;
        }}

        .section {{
            display: none;
            animation: fadeIn 0.5s ease;
        }}

        .section.active {{
            display: block;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .section-title {{
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 8px;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .section-subtitle {{
            color: var(--text-secondary);
            margin-bottom: 32px;
            font-size: 1rem;
        }}

        /* ── Cards ─────────────────────────────────────────────── */
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 28px;
            margin-bottom: 24px;
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-glow);
        }}

        .card h3 {{
            font-size: 1.15rem;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }}

        .card img {{
            width: 100%;
            border-radius: var(--radius-sm);
            margin-top: 12px;
        }}

        /* ── Stats Grid ───────────────────────────────────────── */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 32px;
        }}

        .stat-card {{
            background: var(--gradient-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            text-align: center;
            transition: transform 0.3s ease;
        }}

        .stat-card:hover {{
            transform: scale(1.03);
        }}

        .stat-card .stat-value {{
            font-size: 2rem;
            font-weight: 800;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .stat-card .stat-label {{
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        /* ── Table ─────────────────────────────────────────────── */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
        }}

        th, td {{
            padding: 14px 18px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.06em;
        }}

        tr:hover td {{
            background: rgba(99, 102, 241, 0.05);
        }}

        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}

        .badge.quantum {{
            background: rgba(139, 92, 246, 0.15);
            color: var(--accent-purple);
            border: 1px solid rgba(139, 92, 246, 0.3);
        }}

        .badge.classical {{
            background: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        /* ── Circuit Diagrams ─────────────────────────────────── */
        .circuit-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 20px;
        }}

        .circuit-card h3 {{
            margin-bottom: 12px;
            font-size: 1.05rem;
        }}

        .circuit-diagram {{
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 20px;
            overflow-x: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            line-height: 1.5;
            color: var(--accent-cyan);
        }}

        /* ── Charts ───────────────────────────────────────────── */
        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}

        @media (max-width: 768px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
        }}

        .chart-container {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
        }}

        .chart-container h3 {{
            margin-bottom: 16px;
            font-size: 1.05rem;
        }}

        /* ── Filter Gallery ───────────────────────────────────── */
        .filter-section {{
            margin-bottom: 32px;
        }}

        .filter-section h3 {{
            font-size: 1.1rem;
            margin-bottom: 16px;
            color: var(--accent-cyan);
        }}

        .filter-section img {{
            width: 100%;
            border-radius: var(--radius);
            border: 1px solid var(--border);
        }}

        /* ── Config ───────────────────────────────────────────── */
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 12px;
        }}

        .config-item {{
            display: flex;
            justify-content: space-between;
            padding: 12px 16px;
            background: var(--bg-glass);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
        }}

        .config-item .key {{
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}

        .config-item .val {{
            color: var(--accent-cyan);
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }}

        /* ── Footer ───────────────────────────────────────────── */
        .footer {{
            text-align: center;
            padding: 40px 24px;
            color: var(--text-muted);
            font-size: 0.85rem;
            border-top: 1px solid var(--border);
            margin-top: 60px;
        }}

        /* ── Scrollbar ────────────────────────────────────────── */
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg-primary); }}
        ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: var(--accent-purple); }}
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header" id="top">
        <h1>Quanvolutional Image Filters</h1>
        <p>Hybrid quantum-classical convolutional neural network for MNIST image classification</p>
        <span class="tag">PennyLane + PyTorch &nbsp;·&nbsp; {config['n_qubits']} Qubits &nbsp;·&nbsp; {config['n_epochs']} Epochs</span>
    </div>

    <!-- Navigation -->
    <nav class="nav">
        <button class="active" onclick="showSection('overview')" id="nav-overview">Overview</button>
        <button onclick="showSection('circuits')" id="nav-circuits">Quantum Circuits</button>
        <button onclick="showSection('filters')" id="nav-filters">Filter Gallery</button>
        <button onclick="showSection('training')" id="nav-training">Training</button>
        <button onclick="showSection('config')" id="nav-config">Configuration</button>
    </nav>

    <div class="container">
        <!-- ─── Overview Section ──────────────────────────────── -->
        <div class="section active" id="overview">
            <h2 class="section-title">Model Overview</h2>
            <p class="section-subtitle">Performance summary across all quantum filter architectures</p>

            <div class="stats-grid">
                {"".join(f'''
                <div class="stat-card">
                    <div class="stat-value">{max(hist["val_acc"]):.1%}</div>
                    <div class="stat-label">{name}</div>
                </div>''' for name, hist in zip(names, histories))}
            </div>

            <div class="card">
                <h3>Model Comparison</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Best Accuracy</th>
                            <th>Final Loss</th>
                            <th>Best Epoch</th>
                        </tr>
                    </thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>

            <div class="card">
                <h3>Training Curves</h3>
                {training_img}
            </div>
        </div>

        <!-- ─── Circuits Section ──────────────────────────────── -->
        <div class="section" id="circuits">
            <h2 class="section-title">Quantum Circuit Architectures</h2>
            <p class="section-subtitle">Each circuit processes a 2×2 pixel patch through quantum gates</p>
            {circuit_html}
        </div>

        <!-- ─── Filters Section ───────────────────────────────── -->
        <div class="section" id="filters">
            <h2 class="section-title">Filter Gallery</h2>
            <p class="section-subtitle">Visualizing the effect of each quantum filter on MNIST digits</p>

            {filter_gallery}

            <div class="card">
                <h3>Average Filter Response Heatmaps</h3>
                {heatmap_img}
            </div>
        </div>

        <!-- ─── Training Section ──────────────────────────────── -->
        <div class="section" id="training">
            <h2 class="section-title">Training Metrics</h2>
            <p class="section-subtitle">Interactive charts showing training dynamics</p>

            <div class="chart-grid">
                <div class="chart-container">
                    <h3>Validation Accuracy</h3>
                    <canvas id="accChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Validation Loss</h3>
                    <canvas id="lossChart"></canvas>
                </div>
            </div>
        </div>

        <!-- ─── Config Section ────────────────────────────────── -->
        <div class="section" id="config">
            <h2 class="section-title">Configuration</h2>
            <p class="section-subtitle">Hyperparameters and settings used for this run</p>

            <div class="card">
                <div class="config-grid">
                    {"".join(f'''
                    <div class="config-item">
                        <span class="key">{k}</span>
                        <span class="val">{v}</span>
                    </div>''' for k, v in config.items() if k != 'filter_types')}
                    <div class="config-item">
                        <span class="key">filter_types</span>
                        <span class="val">{', '.join(config['filter_types'])}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer class="footer">
        Quanvolutional Image Filter Dashboard · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} ·
        Based on Henderson et al. (arXiv:1904.04767)
    </footer>

    <script>
        // ── Navigation ──
        function showSection(id) {{
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
            document.getElementById(id).classList.add('active');
            document.getElementById('nav-' + id).classList.add('active');
        }}

        // ── Charts ──
        const labels = {json.dumps(epochs_list)};
        const chartDefaults = {{
            responsive: true,
            plugins: {{
                legend: {{ labels: {{ color: '#94a3b8', font: {{ family: 'Inter' }} }} }},
            }},
            scales: {{
                x: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#64748b' }} }},
                y: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#64748b' }} }},
            }},
        }};

        new Chart(document.getElementById('accChart'), {{
            type: 'line',
            data: {{ labels, datasets: [{acc_datasets}] }},
            options: {{ ...chartDefaults, scales: {{ ...chartDefaults.scales, y: {{ ...chartDefaults.scales.y, min: 0, max: 1 }} }} }},
        }});

        new Chart(document.getElementById('lossChart'), {{
            type: 'line',
            data: {{ labels, datasets: [{loss_datasets}] }},
            options: chartDefaults,
        }});
    </script>
</body>
</html>"""

        path = os.path.join(self.save_dir, "dashboard.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n  [*] Dashboard saved: {os.path.abspath(path)}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("""
+==============================================================+
|          QUANVOLUTIONAL IMAGE FILTER MODEL                   |
|          Hybrid Quantum-Classical CNN - MNIST                |
+==============================================================+
    """)

    config = CONFIG.copy()
    set_seeds(config["seed"])
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ── 1. Load MNIST ─────────────────────────────────────────────────────
    print("\n-- Loading MNIST dataset --")
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Subset
    n_train = config["n_train"]
    n_test = config["n_test"]

    train_images_np = train_data.data[:n_train].numpy().astype(np.float64) / 255.0
    train_labels_np = train_data.targets[:n_train].numpy()
    test_images_np = test_data.data[:n_test].numpy().astype(np.float64) / 255.0
    test_labels_np = test_data.targets[:n_test].numpy()

    print(f"  Train: {n_train} images  |  Test: {n_test} images")

    # ── 2. Build Quantum Filter Bank ──────────────────────────────────────
    print("\n-- Building Quantum Filter Bank --")
    filter_bank = QuantumFilterBank(
        n_qubits=config["n_qubits"],
        n_layers=config["n_layers"],
        seed=config["seed"],
    )

    # Print circuit diagrams
    circuit_diagrams = {}
    for f_name in config["filter_types"]:
        diagram = filter_bank.get_circuit_diagram(f_name)
        circuit_diagrams[f_name] = diagram
        print(f"\n  +-- {f_name} circuit --+")
        for line in diagram.split("\n"):
            print(f"  | {line}")
        print(f"  +{'-' * (len(f_name) + 14)}+")

    # ── 3. Quantum Pre-processing ─────────────────────────────────────────
    print("\n-- Quantum Pre-processing --")
    filtered_train_sets = {}
    filtered_test_sets = {}

    for f_name in config["filter_types"]:
        cache_train = os.path.join(save_dir, f"q_train_{f_name}.npy")
        cache_test = os.path.join(save_dir, f"q_test_{f_name}.npy")

        if config["preprocess"] or not (os.path.exists(cache_train) and os.path.exists(cache_test)):
            print(f"\n  Processing with '{f_name}' filter...")
            circuit_fn = filter_bank.get_filter(f_name)
            engine = QuanvolutionEngine(circuit_fn)

            q_train = engine.process_dataset(train_images_np, label=f"Train ({f_name})")
            q_test = engine.process_dataset(test_images_np, label=f"Test  ({f_name})")

            np.save(cache_train, q_train)
            np.save(cache_test, q_test)
            print(f"  [OK] Cached to {cache_train}")
        else:
            print(f"  Loading cached '{f_name}' data...")
            q_train = np.load(cache_train)
            q_test = np.load(cache_test)

        filtered_train_sets[f_name] = q_train
        filtered_test_sets[f_name] = q_test

    # ── 4. Visualizations ─────────────────────────────────────────────────
    print("\n-- Generating Visualizations --")
    viz = VisualizationGenerator(save_dir)

    viz.plot_quantum_filters(
        train_images_np,
        [filtered_train_sets[n] for n in config["filter_types"]],
        config["filter_types"],
    )

    viz.plot_filter_response_heatmap(
        [filtered_train_sets[n] for n in config["filter_types"]],
        config["filter_types"],
    )

    # ── 5. Training ───────────────────────────────────────────────────────
    print("\n-- Training Phase --")
    all_histories = []
    all_names = []

    # 5a. Classical baseline (raw MNIST images)
    print("\n  Preparing classical baseline...")
    train_tensor = torch.FloatTensor(train_images_np).unsqueeze(1)  # (N, 1, 28, 28)
    test_tensor = torch.FloatTensor(test_images_np).unsqueeze(1)
    train_labels_t = torch.LongTensor(train_labels_np)
    test_labels_t = torch.LongTensor(test_labels_np)

    classical_train_loader = DataLoader(
        TensorDataset(train_tensor, train_labels_t),
        batch_size=config["batch_size"], shuffle=True,
    )
    classical_test_loader = DataLoader(
        TensorDataset(test_tensor, test_labels_t),
        batch_size=config["batch_size"],
    )

    classical_model = ClassicalCNN()
    classical_engine = TrainingEngine(classical_model, device=device, lr=config["learning_rate"])
    classical_hist = classical_engine.fit(
        classical_train_loader, classical_test_loader,
        n_epochs=config["n_epochs"], name="Classical CNN"
    )
    all_histories.append(classical_hist)
    all_names.append("Classical CNN")

    # 5b. Quanvolutional models (one per filter type)
    for f_name in config["filter_types"]:
        q_train = filtered_train_sets[f_name]
        q_test = filtered_test_sets[f_name]

        # Convert (N, H, W, C) → (N, C, H, W) for PyTorch
        q_train_t = torch.FloatTensor(q_train).permute(0, 3, 1, 2)
        q_test_t = torch.FloatTensor(q_test).permute(0, 3, 1, 2)

        q_train_loader = DataLoader(
            TensorDataset(q_train_t, train_labels_t),
            batch_size=config["batch_size"], shuffle=True,
        )
        q_test_loader = DataLoader(
            TensorDataset(q_test_t, test_labels_t),
            batch_size=config["batch_size"],
        )

        img_size = q_train.shape[1]  # 14 for stride=2 on 28×28
        quanv_model = QuanvClassifier(n_channels=config["n_qubits"], img_size=img_size)
        quanv_engine = TrainingEngine(quanv_model, device=device, lr=config["learning_rate"])
        quanv_hist = quanv_engine.fit(
            q_train_loader, q_test_loader,
            n_epochs=config["n_epochs"],
            name=f"Quanv [{f_name}]"
        )
        all_histories.append(quanv_hist)
        all_names.append(f"Quanv [{f_name}]")

    # ── 6. Final Plots & Dashboard ────────────────────────────────────────
    print("\n-- Building Dashboard --")
    viz.plot_training_comparison(all_histories, all_names)

    dashboard_path = viz.generate_html_dashboard(
        all_histories, all_names, config["filter_types"],
        circuit_diagrams, config,
    )

    # Save metrics JSON
    metrics = {}
    for name, hist in zip(all_names, all_histories):
        metrics[name] = {
            "best_val_acc": max(hist["val_acc"]),
            "final_val_loss": hist["val_loss"][-1],
            "best_epoch": hist["val_acc"].index(max(hist["val_acc"])) + 1,
        }
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"""
+==============================================================+
|  [OK]  PIPELINE COMPLETE                                     |
+==============================================================+
|                                                              |
|  Results saved to: {save_dir + '/':40s} |
|  Dashboard:        {os.path.basename(dashboard_path):40s} |
|  Metrics:          metrics.json                              |
|                                                              |
+==============================================================+
    """)

    # Print final summary
    print("\n  Final Results:")
    print("  " + "-" * 50)
    for name, hist in zip(all_names, all_histories):
        best = max(hist["val_acc"])
        print(f"    {name:35s} => {best:.2%} accuracy")
    print("  " + "-" * 50)


if __name__ == "__main__":
    main()

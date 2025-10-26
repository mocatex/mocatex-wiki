import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Modern soft-dark style ---
plt.rcParams.update(
    {
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#262626",
        "axes.edgecolor": "#bbbbbb",
        "axes.labelcolor": "#dddddd",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
        "text.color": "#eeeeee",
        "grid.color": "#444444",
        "grid.linestyle": "--",
        "axes.grid": True,
        "lines.linewidth": 4,
        "font.size": 13,
    }
)

Path("docs/assets/plots/out").mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("docs/assets/plots/out")


##################################################################################
# graph for SIGMOID activation function
x = np.linspace(-10, 10, 200)
y = 1 / (1 + np.exp(-x))
gradient = (np.exp(-x)) / ((1 + np.exp(-x)) ** 2)

plt.axhline(0, color="#eeeeee", linewidth=1)
plt.axvline(0, color="#eeeeee", linewidth=1)
plt.plot(x, y, label="Function")
plt.plot(x, gradient, linestyle="--", color="orange", label="Gradient")
plt.title("Sigmoid Function")
plt.xlabel("Input")
plt.ylabel("Sigmoid(x)")
plt.xticks(np.arange(-10, 11, 2))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid(True)

plt.savefig(OUT_DIR / "sigmoid_activation.svg", bbox_inches="tight")
plt.close()


##################################################################################
# graph for TANH activation function
x = np.linspace(-10, 10, 200)
y = np.tanh(x)
gradient = 1 - np.tanh(x) ** 2

plt.axhline(0, color="#eeeeee", linewidth=1)
plt.axvline(0, color="#eeeeee", linewidth=1)
plt.plot(x, y, label="Function")
plt.plot(x, gradient, linestyle="--", color="orange", label="Gradient")
plt.title("Tanh Function")
plt.xlabel("Input")
plt.ylabel("Tanh(x)")
plt.xticks(np.arange(-10, 11, 2))
plt.yticks(np.arange(-1, 1.1, 0.2))
plt.grid(True)

plt.savefig(OUT_DIR / "tanh_activation.svg", bbox_inches="tight")
plt.close()

##################################################################################
# graph for RELU activation function
x = np.linspace(-5, 5, 200)
y = np.maximum(0, x)
gradient = np.where(x > 0, 1, 0)

plt.axhline(0, color="#eeeeee", linewidth=1)
plt.axvline(0, color="#eeeeee", linewidth=1)
plt.plot(x, y, label="Function")
plt.plot(x, gradient, linestyle="--", color="orange", label="Gradient")
plt.title("ReLU Function")
plt.xlabel("Input")
plt.ylabel("ReLU(x)")
plt.grid(True)

plt.savefig(OUT_DIR / "relu_activation.svg", bbox_inches="tight")
plt.close()

##################################################################################
# graph for LEAKY RELU activation function
x = np.linspace(-5, 5, 200)
y = np.where(x > 0, x, 0.01 * x)
gradient = np.where(x > 0, 1, 0.01)

plt.axhline(0, color="#eeeeee", linewidth=1)
plt.axvline(0, color="#eeeeee", linewidth=1)
plt.plot(x, y, label="Function")
plt.plot(x, gradient, linestyle="--", color="orange", label="Gradient")
plt.title("Leaky ReLU Function")
plt.xlabel("Input")
plt.ylabel("LeakyReLU(x)")
plt.grid(True)

plt.savefig(OUT_DIR / "leaky_relu_activation.svg", bbox_inches="tight")
plt.close()

##################################################################################
# graph for ELU activation function
x = np.linspace(-5, 5, 200)
y = np.where(x > 0, x, np.exp(x) - 1)
gradient = np.where(x > 0, 1, np.exp(x))

plt.axhline(0, color="#eeeeee", linewidth=1)
plt.axvline(0, color="#eeeeee", linewidth=1)
plt.plot(x, y, label="Function")
plt.plot(x, gradient, linestyle="--", color="orange", label="Gradient")
plt.title("ELU Function")
plt.xlabel("Input")
plt.ylabel("ELU(x)")
plt.grid(True)

plt.savefig(OUT_DIR / "elu_activation.svg", bbox_inches="tight")
plt.close()

##################################################################################
# graph for SWISH activation function
x = np.linspace(-5, 5, 200)
y = x / (1 + np.exp(-x))
gradient = (1 / (1 + np.exp(-x))) + (x * np.exp(-x)) / ((1 + np.exp(-x)) ** 2)

plt.axhline(0, color="#eeeeee", linewidth=1)
plt.axvline(0, color="#eeeeee", linewidth=1)
plt.plot(x, y, label="Function")
plt.plot(x, gradient, linestyle="--", color="orange", label="Gradient")
plt.title("Swish Function")
plt.xlabel("Input")
plt.ylabel("Swish(x)")
plt.grid(True)


plt.savefig(OUT_DIR / "swish_activation.svg", bbox_inches="tight")
plt.close()

##################################################################################

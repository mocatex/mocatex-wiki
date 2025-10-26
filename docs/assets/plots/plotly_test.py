import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go

Path("docs/assets/plots/out").mkdir(parents=True, exist_ok=True)
OUT_DIR = Path("docs/assets/plots/out")

# graph for Sigmoid activation function
x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))

fig = go.Figure([go.Scatter(x=x, y=y, mode="lines", name="Sigmoid(x)")])
fig.update_layout(
    xaxis=dict(
        title="Input",
        tickmode="linear",
        dtick=2,
        zeroline=True,
        zerolinewidth=5,
    ),
    yaxis=dict(
        tickmode="linear",
        dtick=0.25,
        title="Sigmoid(x)",
        zeroline=True,
        zerolinewidth=5,
    ),
    margin=dict(t=40),
)

with open(OUT_DIR / "sigmoid_activation.json", "w") as f:
    f.write(fig.to_json() or "")

# add more if you really need a interactive plot otherwise use the matplotlib script

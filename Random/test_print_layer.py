import numpy as np
import plotly.express as px
import torch
import torch.nn as nn

# 1. Create a dummy model (e.g., 100 hidden features, 10 output classes)
hidden_features = 100
num_classes = 10
last_layer = nn.Linear(hidden_features, num_classes)

# 2. Extract weights and convert to a 2D NumPy array
# Shape: (num_classes, hidden_features) -> (10, 100)
weights = last_layer.weight.detach().cpu().numpy()

# 3. Plot using Plotly Express imshow
fig = px.imshow(
    weights,
    labels=dict(x="Hidden Features", y="Output Classes", color="Weight Value"),
    y=[f"Class {i}" for i in range(num_classes)],  # Label the rows
    color_continuous_scale="RdBu_r",  # Red-Blue scale (great for positive/negative weights)
    color_continuous_midpoint=0,  # Centers white at 0
    title="Neural Network Last Layer Weights Visualization",
    aspect="auto",  # Prevents the 10x512 matrix from turning into a skinny strip
)

fig.show()

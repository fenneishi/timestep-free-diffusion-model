import matplotlib.pyplot as plt
import torch
import numpy as np


# Define the function
def custom_function(x):
    x = torch.clamp(700 - x, 0, 700)
    x = x / 700.
    x = x ** 0.25
    x = torch.clamp(x, 0, 1)
    return x


# Generate x values
x_values = torch.linspace(0, 1000, 1000)
# Calculate y values using the custom function
y_values = custom_function(x_values)

area = torch.trapz(y_values, x_values) / 1000
# area_discrete = y_values.sum()

# Convert torch tensors to numpy arrays for plotting
x_values_np = x_values.numpy()
y_values_np = y_values.numpy()

# Plot the curve
plt.figure(figsize=(10, 6))
plt.plot(x_values_np, y_values_np, label='t_signal', color='blue')
plt.title('t = t_signal * t + (1 - t_signal) * torch.randn_like(t)')
plt.xlabel('t')
plt.ylabel('t_signal')
plt.fill_between(x_values_np, y_values_np, alpha=0.3)
plt.text(300, 0.45,
         f"{area * 100:.0f}%",
         horizontalalignment='center',
         verticalalignment='center', fontsize=25,
         color='blue',
         fontname='Comic Sans MS',
         bbox=dict(facecolor='white', alpha=0))
plt.legend()
# plt.grid(True)
plt.savefig('t_signal.png')
plt.close()

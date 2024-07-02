import numpy as np


shape_lib_color_palette = 1. - np.array([
  [0.000, 0.000, 0.545],  # Dark Blue (#00008B) circle
  [0.803, 0.000, 0.000],  # Medium Red (#CD0000) rounged_sq
  [1.000, 0.714, 0.757],  # Light Pink (#FFB6C1) rounded_t_shape
  [0.678, 0.847, 0.902],  # Light Blue (#ADD8E6) rounded_c_shape
  [0.565, 0.933, 0.565],  # Light Green (#90EE90) rounded_i_shape
  [0.941, 0.502, 0.502],  # Light Coral (#F08080) rounded_x_shape
  [1.000, 0.627, 0.478],  # Light Salmon (#FFA07A) rounded_l_shape
  [0.562, 0.000, 0.562],  # Dark Purple. (#8F008F) hexagon
  [0.803, 0.000, 0.803],  # Medium Purple (#CD00CD) pentagon
  [0.803, 0.400, 0.000],  # Dark Orange (#CD6600) slot
  [0.000, 0.803, 0.000],  # Medium Green (#00CD00) rhombus
  [1.000, 0.549, 0.000],   # Medium Orange (#FF8C00) trapezium
  [0.000, 0.000, 0.803],  # Medium Blue (#0000CD) triangle
  [0.545, 0.000, 0.000],  # Dark Red (#8B0000) ellipse_1
  [0.000, 0.392, 0.000],  # Dark Green (#006400) ellipse_2
])


high_res_plot_settings = {
    "figure.dpi": 500,  # High resolution for print
    "savefig.dpi": 300,  # Resolution when saving figures
    "font.family": "serif",  # Serif fonts are often preferred in publications
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": [6, 4],  # Adjust for your specific needs
    "text.usetex": False,  # Set to True if you want LaTeX rendering
    "lines.linewidth": 1.5,  
    "axes.linewidth": 1.0,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "savefig.bbox": "tight",  # Ensures no excessive white space
    "savefig.format": "pdf",  # PDF is a good format for publications
}
########################################################################################################################
#___________                      __________.__          __       _____                .__                             # 
#\__    ___/____   __ __   ____   \______   \  |   _____/  |_    /  _  \   ____ _____  |  | ___.__. ______ ___________ # 
#  |    |  \__  \ |  |  \_/ ___\   |     ___/  |  /  _ \   __\  /  /_\  \ /    \\__  \ |  |<   |  |/  ___// __ \_  __ \# 
#  |    |   / __ \|  |  /\  \___   |    |   |  |_(  <_> )  |   /    |    \   |  \/ __ \|  |_\___  |\___ \\  ___/|  | \/# 
#  |____|  (____  /____/  \___  >  |____|   |____/\____/|__|   \____|__  /___|  (____  /____/ ____/____  >\___  >__|   # 
#               \/            \/                                       \/     \/     \/     \/         \/     \/       # 
#                                               Author: Axel Tracol Gavard                                             #
#                                                                                                                      #
#                                                Last updated: 17.09.2025                                              #
########################################################################################################################

import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename

def import_absorption_data():
    global data

    f_types = [('CSV Files', '*.csv'), ('text Files', '*.txt')]

    file_path = askopenfilename(filetypes=f_types)
    if not file_path:
        return
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, skiprows=5, header=None)
        num_cols = df.shape[1]
        columns = df.iloc[0]
        cleaned_data = {}
        wavelengths = None
        for i in range(0, num_cols, 2):
            if i + 1 < num_cols:
                label = columns[i + 1]
                if label in []:
                    continue
                wavelengths_col = df.iloc[1:, i].astype(float)
                values = df.iloc[1:, i + 1].astype(float)
                if wavelengths is None:
                    wavelengths = wavelengths_col.values
                cleaned_data[label] = values.values
        data = pd.DataFrame(cleaned_data, index=wavelengths)
        # Update the sample dropdown menu
        sample_menu['menu'].delete(0, 'end')
        for sample in data.columns:
            sample_menu['menu'].add_command(label=sample, command=tk._setit(sample_var, sample))
        sample_var.set(data.columns[0])  # Set default selection
    elif file_path.endswith('.txt'):
        df = pd.read_csv(file_path, sep=",", skiprows=2, header=None, names=["Wavelength", "Values"])
        cleaned_df = df.iloc[0::2]
        cleaned_data = {}
        wavelengths = None
        if wavelengths is None:
            wavelengths = cleaned_df["Wavelength"].values.astype(float)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        cleaned_data[file_name] = cleaned_df["Values"].values.astype(float)
        data = pd.DataFrame(cleaned_data, index=wavelengths)

        sample_menu["menu"].delete(0, 'end')
        sample_menu["menu"].add_command(label=file_name, command=tk._setit(sample_var, file_name))
        sample_var.set(data.columns[0])

    # Calculate min and max photon energy
    photon_energy = energy(data.index)
    min_e = min(photon_energy)
    max_e = max(photon_energy)

    # Update sliders' ranges and values
    lb_1_slider.config(from_=min_e, to=max_e)
    ub_1_slider.config(from_=min_e, to=max_e)
    lb_2_slider.config(from_=min_e, to=max_e)
    ub_2_slider.config(from_=min_e, to=max_e)

    # Set default slider values (e.g., 20% and 80% of the range)
    lb_1_var.set(min_e + 0.2 * (max_e - min_e))
    ub_1_var.set(min_e + 0.8 * (max_e - min_e))
    lb_2_var.set(min_e + 0.2 * (max_e - min_e))
    ub_2_var.set(min_e + 0.8 * (max_e - min_e))

    return data

def import_t_r_data():
    global data, T, R

    f_types = [('CSV Files', '*.csv'), ('text Files', '*.txt')]

    r_file_path = askopenfilename(filetypes=f_types, title="Select Reflectance File")
    if not r_file_path:
        return

    t_file_path = askopenfilename(filetypes=f_types, title="Select Transmittance File")
    if not t_file_path:
        return

    # Load Reflectance data
    if r_file_path.endswith('.csv'):
        r_df = pd.read_csv(r_file_path, skiprows=5, header=None)
        num_cols = r_df.shape[1]
        r_columns = r_df.iloc[0]
        r_cleaned_data = {}
        wavelengths = None
        for i in range(0, num_cols, 2):
            if i + 1 < num_cols:
                label = r_columns[i + 1]
                if label in []:
                    continue
                wavelengths_col = r_df.iloc[1:, i].astype(float)
                values = r_df.iloc[1:, i + 1].astype(float)
                if wavelengths is None:
                    wavelengths = wavelengths_col.values
                r_cleaned_data[label] = values.values
        R = pd.DataFrame(r_cleaned_data, index=wavelengths)
    elif r_file_path.endswith('.txt'):
        r_df = pd.read_csv(r_file_path, sep=",", skiprows=2, header=None, names=["Wavelength", "Values"])
        r_cleaned_df = r_df.iloc[0::2]
        r_cleaned_data = {}
        wavelengths = None
        if wavelengths is None:
            wavelengths = r_cleaned_df["Wavelength"].values.astype(float)
        file_name = os.path.splitext(os.path.basename(r_file_path))[0]
        r_cleaned_data[file_name] = r_cleaned_df["Values"].values.astype(float)
        R = pd.DataFrame(r_cleaned_data, index=wavelengths)

    # Load Transmittance data
    if t_file_path.endswith('.csv'):
        t_df = pd.read_csv(t_file_path, skiprows=5, header=None)
        num_cols = t_df.shape[1]
        t_columns = t_df.iloc[0]
        t_cleaned_data = {}
        wavelengths = None
        for i in range(0, num_cols, 2):
            if i + 1 < num_cols:
                label = t_columns[i + 1]
                if label in []:
                    continue
                wavelengths_col = t_df.iloc[1:, i].astype(float)
                values = t_df.iloc[1:, i + 1].astype(float)
                if wavelengths is None:
                    wavelengths = wavelengths_col.values
                t_cleaned_data[label] = values.values
        T = pd.DataFrame(t_cleaned_data, index=wavelengths)
    elif t_file_path.endswith('.txt'):
        t_df = pd.read_csv(t_file_path, sep=",", skiprows=2, header=None, names=["Wavelength", "Values"])
        t_cleaned_df = t_df.iloc[0::2]
        t_cleaned_data = {}
        wavelengths = None
        if wavelengths is None:
            wavelengths = t_cleaned_df["Wavelength"].values.astype(float)
        file_name = os.path.splitext(os.path.basename(t_file_path))[0]
        t_cleaned_data[file_name] = t_cleaned_df["Values"].values.astype(float)
        T = pd.DataFrame(t_cleaned_data, index=wavelengths)

    # Merge R and T into a single DataFrame
    data = pd.concat([R, T], axis=1)

    # Update the sample dropdown menu
    sample_menu['menu'].delete(0, 'end')
    for sample in data.columns:
        sample_menu['menu'].add_command(label=sample, command=tk._setit(sample_var, sample))
    sample_var.set(data.columns[0])  # Set default selection

    # Calculate min and max photon energy
    photon_energy = energy(data.index)
    min_e = min(photon_energy)
    max_e = max(photon_energy)

    # Update sliders' ranges and values
    lb_1_slider.config(from_=min_e, to=max_e)
    ub_1_slider.config(from_=min_e, to=max_e)
    lb_2_slider.config(from_=min_e, to=max_e)
    ub_2_slider.config(from_=min_e, to=max_e)

    # Set default slider values (e.g., 20% and 80% of the range)
    lb_1_var.set(min_e + 0.2 * (max_e - min_e))
    ub_1_var.set(min_e + 0.8 * (max_e - min_e))
    lb_2_var.set(min_e + 0.2 * (max_e - min_e))
    ub_2_var.set(min_e + 0.8 * (max_e - min_e))

    print(data)
    return data

def energy(data):
    photon_energy = np.array(1240 / data)
    return photon_energy

def calculate_tauc(ab_coeff, energy, n):
    return np.array((ab_coeff * energy)**n)

def line_intersection(line1x, line1y, line2x, line2y):
    """
    Find the intersection of two lines defined by arrays of points.

    Parameters:
    - line1x, line1y: Arrays of x and y coordinates for the first line
    - line2x, line2y: Arrays of x and y coordinates for the second line

    Returns:
    - x, y: Coordinates of the intersection point
    """
    # Convert arrays to numpy arrays if they're not already
    line1x = np.array(line1x)
    line1y = np.array(line1y)
    line2x = np.array(line2x)
    line2y = np.array(line2y)

    # Fit lines to get slope and intercept
    coeffs1 = np.polyfit(line1x, line1y, 1)
    coeffs2 = np.polyfit(line2x, line2y, 1)

    # Extract slope and intercept
    m1, b1 = coeffs1
    m2, b2 = coeffs2

    # Check if lines are parallel
    if m1 == m2:
        raise Exception('Lines are parallel, they do not intersect')

    # Calculate intersection point
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1

    return x, y

def plot_tauc(_=None):
    sample = sample_var.get()
    spc_type = spc_var.get()
    transition_type = n_var.get()
    distance = d_var.get()

    if spc_type == "Absorbance":
        abs_coeff = data[sample]/distance
    else:
        abs_coeff = -np.log(T[sample]/(100 - R[sample]))/distance

    if transition_type == "Direct allowed (n=2)":
        n = 2
        tauc_values = calculate_tauc(abs_coeff, energy(data.index), n=n)
        y_label = r"$(\alpha E)^2$"
    elif transition_type == "Indirect allowed (n=1/2)":
        n = 0.5
        tauc_values = calculate_tauc(abs_coeff, energy(data.index), n=n)
        y_label = r"$(\alpha E)^{1/2}$"
    elif transition_type == "Direct forbidden (n=2/3)":
        n = 2/3
        tauc_values = calculate_tauc(abs_coeff, energy(data.index), n=n)
        y_label = r"$(\alpha E)^{2/3}$"
    else:
        n = 1/3
        tauc_values = calculate_tauc(abs_coeff, energy(data.index), n=n)
        y_label = r"$(\alpha E)^{1/3}$"

    ax.clear()
    ax.plot(energy(data.index), tauc_values,
              color='black', linewidth=1.5, alpha=0.7,
              label='Full spectrum')
    
    # Find indices corresponding to the energy bounds
    indices_1 = np.where((energy(data.index) >= lb_1_slider.get()) & (energy(data.index) <= ub_1_slider.get()))[0]

    if len(indices_1) > 0:
        x_fit_1 = energy(data.index)[indices_1]
        y_fit_1 = tauc_values[indices_1]
        
        ax.plot(x_fit_1, y_fit_1, 'b-', linewidth=2.5, label='Selected range 1')
        
        coeffs_1 = np.polyfit(x_fit_1, y_fit_1, 1)
        poly_1 = np.poly1d(coeffs_1)
        
        x_line_1 = np.linspace(lb_1_slider.get() * 0.7, ub_1_slider.get() * 1.3, 100)
        ax.plot(x_line_1, poly_1(x_line_1), 'g--', linewidth=1.5, 
                    label=f'Linear fit 1: y = {coeffs_1[0]:.2f}x {coeffs_1[1] :+.2f}')
        
    # Find indices corresponding to the energy bounds
    indices_2 = np.where((energy(data.index) >= lb_2_slider.get()) & (energy(data.index) <= ub_2_slider.get()))[0]

    if len(indices_2) > 0:
        x_fit_2 = energy(data.index)[indices_2]
        y_fit_2 = tauc_values[indices_2]
        
        ax.plot(x_fit_2, y_fit_2, 'r-', linewidth=2.5, label='Selected range 2')
        
        coeffs_2 = np.polyfit(x_fit_2, y_fit_2, 1)
        poly_2 = np.poly1d(coeffs_2)
        
        x_line_2 = np.linspace(lb_2_slider.get() * 0.7, ub_2_slider.get() * 1.3, 100)
        ax.plot(x_line_2, poly_2(x_line_2), 'g--', linewidth=1.5, 
                    label=f'Linear fit 2: y = {coeffs_2[0]:.2f}x {coeffs_2[1] :+.2f}')

    x, y = line_intersection(line1x=x_line_1, line1y=poly_1(x_line_1),
                              line2x=x_line_2, line2y=poly_2(x_line_2))
    bandgap_var.set(f"Bandgap: {x:.3f} eV")
    
    ax.set_xlabel("Photon Energy (eV)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Tauc Plot for {sample} - {transition_type}")
    ax.legend(loc='upper left')
    
    # Set appropriate axes limits
    ax.set_xlim(min(energy(data.index)) * 0.9, max(energy(data.index)) * 1.1)
    
    # Display the plot
    canvas.draw()

### Build app GUI ###
root = tk.Tk(className= "Tauc Plot Analyser")
bandgap_var = tk.StringVar(value="Bandgap: -- eV")

paned1 = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
paned1.pack(fill=tk.BOTH, expand=True)

plot_frame = ttk.Frame(paned1, padding=5)
paned1.add(plot_frame, weight=3)

ctrl_frame = ttk.LabelFrame(paned1, text="Controls", padding=10)
paned1.add(ctrl_frame, weight=1)

fig = Figure(figsize=(5, 5), dpi=100)
ax = fig.add_subplot(111)   # primary axis

canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

toolbar = NavigationToolbar2Tk(canvas, plot_frame)
toolbar.update()
toolbar.pack(side=tk.BOTTOM, fill=tk.X)

import_button_ab = ttk.Button( 
    ctrl_frame, text="Import Absorption Data", command=import_absorption_data
)
import_button_ab.pack(fill=tk.X, pady=5)

import_button_t_r = ttk.Button(
    ctrl_frame, text="Import Transmittance and Reflectance Data", command=import_t_r_data
    )
import_button_t_r.pack(fill=tk.X, pady=5)

d_label = ttk.Label(ctrl_frame, text="Distance (cm)")
d_label.pack(fill=tk.X, pady=5)
d_var = tk.DoubleVar()
d_input = ttk.Entry(ctrl_frame, textvariable=d_var)
d_input.pack(fill=tk.X, pady=5)

spc_var = tk.StringVar(value="Absorbance")
spc_menu = ttk.OptionMenu(ctrl_frame, spc_var,
            "Absorbance",
            *["Absorbance", "Transmittance + Reflectance"])
spc_menu.pack(fill=tk.X, pady=5)

n_var = tk.StringVar(value="Direct allowed (n=2)")
n_menu = ttk.OptionMenu(ctrl_frame, n_var,
        "Direct allowed (n=2)", 
        *["Direct allowed (n=2)", "Indirect allowed (n=1/2)", "Direct forbidden (n=2/3)", "Indirect forbidden (n=1/3)"],
        command=plot_tauc
        )
n_menu.pack(fill=tk.X, pady=5)

sample_var = tk.StringVar()
sample_menu = ttk.OptionMenu(
    ctrl_frame, sample_var, "",
    command=plot_tauc
)
sample_menu.pack(fill=tk.X, pady=5)

lb_1_var = tk.DoubleVar(value=2)
ub_1_var = tk.DoubleVar(value=3)
lb_2_var = tk.DoubleVar(value=6)
ub_2_var = tk.DoubleVar(value=7)

lb_1_label = ttk.Label(ctrl_frame, text="Lower Bound 1 (eV)")
lb_1_label.pack(fill=tk.X, padx=(0, 5), pady=(10, 0))
lb_1_val_label = ttk.Label(ctrl_frame, textvariable=lb_1_var, anchor="w")
lb_1_val_label.pack(fill=tk.X, pady=0)
lb_1_slider = ttk.Scale(
    ctrl_frame, 
    from_=0, 
    to=10, orient='horizontal',
    variable=lb_1_var, command=plot_tauc
)
lb_1_slider.pack(fill=tk.X, pady=(0, 10))

ub_1_label = ttk.Label(ctrl_frame, text="Upper Bound 1 (eV)")
ub_1_label.pack(fill=tk.X, padx=(0,5), pady=(10, 0))
ub_1_val_label = ttk.Label(ctrl_frame, textvariable=ub_1_var, anchor="w")
ub_1_val_label.pack(fill=tk.X, pady=0)
ub_1_slider = ttk.Scale(
    ctrl_frame, from_=0, 
    to=10, orient='horizontal',
    variable=ub_1_var, command=plot_tauc
)
ub_1_slider.pack(fill=tk.X, pady=(0, 10))

lb_2_label = ttk.Label(ctrl_frame, text="Lower Bound 2 (eV)")
lb_2_label.pack(fill=tk.X, padx=(0,5), pady=(10, 0))
lb_2_val_label = ttk.Label(ctrl_frame, textvariable=lb_2_var, anchor="w")
lb_2_val_label.pack(fill=tk.X, pady=0)
lb_2_slider = ttk.Scale(
    ctrl_frame, from_=0, 
    to=10, orient='horizontal',
    variable=lb_2_var, command=plot_tauc
)
lb_2_slider.pack(fill=tk.X, pady=(0, 10))

ub_2_label = ttk.Label(ctrl_frame, text="Upper Bound 2 (eV)")
ub_2_label.pack(fill=tk.X, padx=(0,5), pady=0)
ub_2_val_label = ttk.Label(ctrl_frame, textvariable=ub_2_var, anchor="w")
ub_2_val_label.pack(fill=tk.X, pady=(10,0))
ub_2_slider = ttk.Scale(
    ctrl_frame, from_=0, 
    to=10, orient='horizontal',
    variable=ub_2_var, command=plot_tauc
)
ub_2_slider.pack(fill=tk.X, pady=(0, 10))

bandgap_label = ttk.Label(ctrl_frame, textvariable=bandgap_var, relief="groove", anchor="center")
bandgap_label.pack(fill=tk.X, pady=5)

root.mainloop()








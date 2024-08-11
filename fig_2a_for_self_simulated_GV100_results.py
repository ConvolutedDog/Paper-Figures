
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import matplotlib.ticker as ticker

import pandas as pd

np.random.seed(19680801)


Results_xlsx_path = "../Self-Simulated-Results/compare_for_self_simulated_GV100_results.xlsx"


def plot_scatter_x_HW_Cycles_y_Simulated_Cycles(ax, indexes, cycles, marker="", markersize=1, color="", label=""):
    """Scatter plot."""
    
    indexes_keys = [_ for _ in indexes.keys()]
    x = [indexes[_] for _ in indexes_keys]
    y = [cycles[_] for _ in indexes_keys]

    x_numpy = np.array(x)
    y_numpy = np.array(y)

    def log(x, base=10):
        return np.log(x)/np.log(base)

    max_error = max(log(y_numpy) - log(x_numpy))
    min_error = min(log(y_numpy) - log(x_numpy))

    ax.plot(x, y, ls='none', marker=marker, color=color, markersize=markersize, label=label)
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlim(min(x_numpy),max(x_numpy))
    ax.set_xlabel('HW Execution Time (s)', fontsize=30)
    ax.set_ylabel('Simulation Time (s)', fontsize=30)
    ax.set_title('')
    return ax

def read_xlsx_GPU_simulation_time(file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_simulation_time = {}
    PPT_simulation_time = {}
    ASIM_simulation_time = {}
    OURS_simulation_time = {}

    None_of_OURS_simulation_time = []

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_PPT = pd.read_excel(file_name, sheet_name=PPT_sheet_name)
    data_ASIM = pd.read_excel(file_name, sheet_name=ASIM_sheet_name)
    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_simulation_time = row["Memory model time (s)"] + row["Compute model time (s)"]
        if 1:
            if not kernel_key in OURS_simulation_time.keys():
                if isinstance(kernel_simulation_time, (int, float)) and not pd.isna(kernel_simulation_time):
                    OURS_simulation_time[kernel_key] = kernel_simulation_time
                else:
                    None_of_OURS_simulation_time.append({kernel_key: kernel_id})
            else:
                if isinstance(kernel_simulation_time, (int, float)) and not pd.isna(kernel_simulation_time):
                    OURS_simulation_time[kernel_key] += kernel_simulation_time
                else:
                    None_of_OURS_simulation_time.append({kernel_key: kernel_id})

    for idx, row in data_ASIM.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_simulation_time = row["Unnamed: 34"]
        if not {kernel_key: kernel_id} in None_of_OURS_simulation_time:
            if not kernel_key in ASIM_simulation_time.keys():
                if isinstance(kernel_simulation_time, (int, float)):
                    ASIM_simulation_time[kernel_key] = kernel_simulation_time
            else:
                if isinstance(kernel_simulation_time, (int, float)):
                    ASIM_simulation_time[kernel_key] += kernel_simulation_time

    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_simulation_time = row["Kernel execution time (ns)"] * 1e-9
        if not {kernel_key: kernel_id} in None_of_OURS_simulation_time:
            if kernel_key in ASIM_simulation_time.keys():
                if not kernel_key in NCU_simulation_time.keys():
                    if isinstance(kernel_simulation_time, (int, float)):
                        NCU_simulation_time[kernel_key] = kernel_simulation_time
                else:
                    if isinstance(kernel_simulation_time, (int, float)):
                        NCU_simulation_time[kernel_key] += kernel_simulation_time

    for idx, row in data_PPT.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_simulation_time = row["Unnamed: 34"] + row["Unnamed: 35"]
        if not {kernel_key: kernel_id} in None_of_OURS_simulation_time:
            if kernel_key in ASIM_simulation_time.keys():
                if not kernel_key in PPT_simulation_time.keys():
                    if isinstance(kernel_simulation_time, (int, float)):
                        PPT_simulation_time[kernel_key] = kernel_simulation_time
                else:
                    if isinstance(kernel_simulation_time, (int, float)):
                        PPT_simulation_time[kernel_key] += kernel_simulation_time

    def seconds_to_dhms(seconds):
        days = seconds // (24 * 3600)
        seconds = seconds % (24 * 3600)
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60

        result = ""
        if days > 0:
            result += f"{days}d "
        if hours > 0:
            result += f"{hours}h "
        if minutes > 0:
            result += f"{minutes}m "
        if seconds > 0 or (days == 0 and hours == 0 and minutes == 0):
            result += f"{seconds}s"

        return result.strip()

    ASIM_simulation_time_total = 0
    PPT_simulation_time_total = 0
    OURS_simulation_time_total = 0
    for key in NCU_simulation_time.keys():
        if key in ASIM_simulation_time.keys() and key in PPT_simulation_time.keys() and \
           key in OURS_simulation_time.keys():
            if key == "gaussian" or key == "lud":
                continue
            ASIM_simulation_time_total += ASIM_simulation_time[key]
            PPT_simulation_time_total += PPT_simulation_time[key]
            OURS_simulation_time_total += OURS_simulation_time[key]
    print("Total: ")
    print("  ASIM_simulation_time_total: ", seconds_to_dhms(ASIM_simulation_time_total))
    print("  PPT_simulation_time_total: ", seconds_to_dhms(PPT_simulation_time_total))
    print("  OURS_simulation_time_total: ", seconds_to_dhms(OURS_simulation_time_total))
    print("")
    print("  OURS is", float(ASIM_simulation_time_total) / float(OURS_simulation_time_total), "X faster than ASIM")
    print("  OURS is", 1.0 - float(OURS_simulation_time_total) / float(PPT_simulation_time_total), "X slower than PPT")

    return NCU_simulation_time, PPT_simulation_time, ASIM_simulation_time, OURS_simulation_time



def plot_figure_GPU_Simulation_Time_Slowdown(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_GPU_simulation_time(
                Results_xlsx_path,
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="OURS")
    
    NCU_simulation_time, PPT_simulation_time, \
    ASIM_simulation_time, OURS_simulation_time = prng[0], prng[1], prng[2], prng[3]

    NCU_simulation_time = {k: v \
                           for k, v in sorted(NCU_simulation_time.items(), key=lambda item: item[1], reverse=False)}
    PPT_simulation_time = {k: PPT_simulation_time[k] \
                           for k, v in sorted(NCU_simulation_time.items(), key=lambda item: item[1], reverse=False)}
    ASIM_simulation_time = {k: ASIM_simulation_time[k] \
                           for k, v in sorted(NCU_simulation_time.items(), key=lambda item: item[1], reverse=False)}
    OURS_simulation_time = {k: OURS_simulation_time[k] \
                           for k, v in sorted(NCU_simulation_time.items(), key=lambda item: item[1], reverse=False)}


    indexes = list(NCU_simulation_time.keys())

    fig, axs = plt.subplots(ncols=1, nrows=1, num=style_label,
                            figsize=(7.8, 7.8), layout='constrained')
    
    axs.plot(NCU_simulation_time.values(), NCU_simulation_time.values(), \
             ls='--', color='#949494', linewidth=5, label="")
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_simulation_time, NCU_simulation_time, \
                                                marker='^', markersize=20, color="#c387c3", label="NCU")
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_simulation_time, PPT_simulation_time, \
                                                marker='s', markersize=20, color="#fcca99", label="PPT")
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_simulation_time, ASIM_simulation_time, \
                                                marker='o', markersize=20, color="#8ad9f8", label="ASIM")
    plot_scatter_x_HW_Cycles_y_Simulated_Cycles(axs, NCU_simulation_time, OURS_simulation_time, \
                                                marker='P', markersize=20, color="pink", label="HyFiSS")
    axs.legend(loc='lower right', fontsize=25, frameon=True, shadow=True, fancybox=False, framealpha=1.0, borderpad=0.3,
           ncol=1, markerfirst=True, markerscale=1.3, numpoints=1, handlelength=2.0)

    # axs.set_ylim(0, 1e6-1)

    axs.grid(True, which='major', axis='both', linestyle='--', color='gray', linewidth=1)


if __name__ == "__main__":
    style_list = ['classic']

    for style_label in style_list:
        with plt.rc_context({"figure.max_open_warning": len(style_list)}):
            with plt.style.context(style_label):
                plot_figure_GPU_Simulation_Time_Slowdown(style_label=style_label)
                plt.savefig('figs/'+'fig_2a_for_self_simulated_GV100_results.pdf', format='pdf')


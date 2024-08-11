
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
plt.rcParams['hatch.color'] = 'white'
import matplotlib
# matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
plt.rcParams['hatch.linewidth'] = 0.1
import numpy as np

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import matplotlib.ticker as ticker

import pandas as pd

Results_xlsx_path = "../Self-Simulated-Results/compare_for_self_simulated_GV100_results.xlsx"

np.random.seed(19680801)

indexes_short_name_dict = {
    "2DConvolution": "2DConv",
    "3DConvolution": "3DConv",
    "cublas_GemmEx_HF_TC_example_128x128x128": "TGEMMx128",
    "cublas_GemmEx_HF_TC_example_256x256x256": "TGEMMx256",
    "cublas_GemmEx_HF_TC_example_512x512x512": "TGEMMx512",
    "cublas_GemmEx_HF_TC_example_1024x1024x1024": "TGEMMx1024",
    "cublas_GemmEx_HF_TC_example_2048x2048x2048": "TGEMMx2048",
    "cublas_GemmEx_HF_TC_example_4096x4096x4096": "TGEMMx4096",
    "cublas_GemmEx_HF_CC_example_128x128x128": "CGEMMx128",
    "cublas_GemmEx_HF_CC_example_256x256x256": "CGEMMx256",
    "cublas_GemmEx_HF_CC_example_512x512x512": "CGEMMx512",
    "cublas_GemmEx_HF_CC_example_1024x1024x1024": "CGEMMx1024",
    "cublas_GemmEx_HF_CC_example_2048x2048x2048": "GemmEx",
    "cublas_GemmEx_HF_CC_example_4096x4096x4096": "CGEMMx4096",
    "conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2": "conv_inf",
    "conv_bench_train_halfx700x161x1x1x32x20x5x0x0x2x2": "conv_train",
    "gemm_bench_inference_halfx1760x7000x1760x0x0": "gemm_inf",
    "gemm_bench_train_halfx1760x7000x1760x0x0": "gemm_train",
    "rnn_bench_inference_halfx1024x1x25xlstm": "rnn_inf",
    "rnn_bench_train_halfx1024x1x25xlstm": "rnn_train",
    "lulesh": "Lulesh",
    "pennant": "Pennant",
    "b+tree": "b+tree",
    "backprop": "backprop",
    "bfs": "bfs",
    "dwt2d": "dwt2d",
    "gaussian": "gaussian",
    "hotspot": "hotspot",
    "huffman": "huffman",
    "lavaMD": "lavaMD",
    "nn": "nn",
    "pathfinder": "pathfinder",
    "2DConvolution": "2DConv",
    "3DConvolution": "3DConv",
    "3mm": "3mm",
    "atax": "atax",
    "bicg": "bicg",
    "gemm": "gemm",
    "gesummv": "gesummv",
    "gramschmidt": "gramsch",
    "mvt": "mvt",
    "cfd": "cfd",
    "hotspot3D": "hotspot3D",
    "lud": "lud",
    "nw": "nw",
    "AN_32": "AlexNet",
    "GRU": "GRU",
    "LSTM_32": "LSTM",
    "SN_32": "SqueezeNet",
}

Except_keys = [
    "cublas_GemmEx_HF_TC_example_128x128x128",
    "cublas_GemmEx_HF_TC_example_256x256x256",
    "cublas_GemmEx_HF_TC_example_512x512x512",
    "cublas_GemmEx_HF_CC_example_1024x1024x1024",
    "cublas_GemmEx_HF_CC_example_128x128x128",
    "cublas_GemmEx_HF_CC_example_256x256x256",
    "cublas_GemmEx_HF_CC_example_512x512x512",
]

def read_xlsx_MemData_Stalls(file_name="", OURS_sheet_name=""):
    
    MemoryDataStall_Issue_scoreboard_Cycles = {}
    MemoryDataStall_Execute_L1_Cycles = {}
    MemoryDataStall_Execute_L2_Cycles = {}
    MemoryDataStall_Execute_Main_Memory_Cycles = {}
    
    None_of_OURS_System_Stalls = []

    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        
        kernel_MemoryDataStall_Issue_scoreboard_Cycles = row["MemoryDataStall_Issue_scoreboard_Cycles"]
        kernel_MemoryDataStall_Execute_L1_Cycles = row["MemoryDataStall_Execute_L1_Cycles"]
        kernel_MemoryDataStall_Execute_L2_Cycles = row["MemoryDataStall_Execute_L2_Cycles"]
        kernel_MemoryDataStall_Execute_Main_Memory_Cycles = row["MemoryDataStall_Execute_Main_Memory_Cycles"]
        
        if not kernel_key in Except_keys:
            if not kernel_key in MemoryDataStall_Issue_scoreboard_Cycles.keys():
                if isinstance(kernel_MemoryDataStall_Issue_scoreboard_Cycles, (int, float)) and \
                    not pd.isna(kernel_MemoryDataStall_Issue_scoreboard_Cycles) and \
                    isinstance(kernel_MemoryDataStall_Execute_L1_Cycles, (int, float)) and \
                    not pd.isna(kernel_MemoryDataStall_Execute_L1_Cycles) and \
                    isinstance(kernel_MemoryDataStall_Execute_L2_Cycles, (int, float)) and \
                    not pd.isna(kernel_MemoryDataStall_Execute_L2_Cycles) and \
                    isinstance(kernel_MemoryDataStall_Execute_Main_Memory_Cycles, (int, float)) and \
                    not pd.isna(kernel_MemoryDataStall_Execute_Main_Memory_Cycles):
                    
                    MemoryDataStall_Issue_scoreboard_Cycles[kernel_key] = kernel_MemoryDataStall_Issue_scoreboard_Cycles
                    MemoryDataStall_Execute_L1_Cycles[kernel_key] = kernel_MemoryDataStall_Execute_L1_Cycles
                    MemoryDataStall_Execute_L2_Cycles[kernel_key] = kernel_MemoryDataStall_Execute_L2_Cycles
                    MemoryDataStall_Execute_Main_Memory_Cycles[kernel_key] = kernel_MemoryDataStall_Execute_Main_Memory_Cycles
                else:
                    continue
            else:
                if isinstance(kernel_MemoryDataStall_Issue_scoreboard_Cycles, (int, float)) and \
                    not pd.isna(kernel_MemoryDataStall_Issue_scoreboard_Cycles) and \
                    isinstance(kernel_MemoryDataStall_Execute_L1_Cycles, (int, float)) and \
                    not pd.isna(kernel_MemoryDataStall_Execute_L1_Cycles) and \
                    isinstance(kernel_MemoryDataStall_Execute_L2_Cycles, (int, float)) and \
                    not pd.isna(kernel_MemoryDataStall_Execute_L2_Cycles) and \
                    isinstance(kernel_MemoryDataStall_Execute_Main_Memory_Cycles, (int, float)) and \
                    not pd.isna(kernel_MemoryDataStall_Execute_Main_Memory_Cycles):

                    MemoryDataStall_Issue_scoreboard_Cycles[kernel_key] += kernel_MemoryDataStall_Issue_scoreboard_Cycles
                    MemoryDataStall_Execute_L1_Cycles[kernel_key] += kernel_MemoryDataStall_Execute_L1_Cycles
                    MemoryDataStall_Execute_L2_Cycles[kernel_key] += kernel_MemoryDataStall_Execute_L2_Cycles
                    MemoryDataStall_Execute_Main_Memory_Cycles[kernel_key] += kernel_MemoryDataStall_Execute_Main_Memory_Cycles
                else:
                    continue
        else:
            continue
    
    MemoryDataStall_Issue_scoreboard_Cycles_rate = {}
    MemoryDataStall_Execute_L1_Cycles_rate = {}
    MemoryDataStall_Execute_L2_Cycles_rate = {}
    MemoryDataStall_Execute_Main_Memory_Cycles_rate = {}

    for key in MemoryDataStall_Issue_scoreboard_Cycles.keys():
        all_cycles = float(MemoryDataStall_Issue_scoreboard_Cycles[key] + MemoryDataStall_Execute_L1_Cycles[key] + \
            MemoryDataStall_Execute_L2_Cycles[key] + MemoryDataStall_Execute_Main_Memory_Cycles[key])
        MemoryDataStall_Issue_scoreboard_Cycles_rate[key] = \
            float(MemoryDataStall_Issue_scoreboard_Cycles[key]) / all_cycles
        MemoryDataStall_Execute_L1_Cycles_rate[key] = \
            float(MemoryDataStall_Execute_L1_Cycles[key]) / all_cycles
        MemoryDataStall_Execute_L2_Cycles_rate[key] = \
            float(MemoryDataStall_Execute_L2_Cycles[key]) / all_cycles
        MemoryDataStall_Execute_Main_Memory_Cycles_rate[key] = \
            float(MemoryDataStall_Execute_Main_Memory_Cycles[key]) / all_cycles
        


    return MemoryDataStall_Issue_scoreboard_Cycles_rate, MemoryDataStall_Execute_L1_Cycles_rate, \
           MemoryDataStall_Execute_L2_Cycles_rate, MemoryDataStall_Execute_Main_Memory_Cycles_rate
           


def plot_figure_StackBar_System_Stalls(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_MemData_Stalls(
                Results_xlsx_path, 
                OURS_sheet_name="OURS")
    
    OURS_DATA = None

    MemoryDataStall_Issue_scoreboard_Cycles_rate, MemoryDataStall_Execute_L1_Cycles_rate, \
    MemoryDataStall_Execute_L2_Cycles_rate, MemoryDataStall_Execute_Main_Memory_Cycles_rate = \
    prng[0], prng[1], prng[2], prng[3]

    indexes = list(MemoryDataStall_Issue_scoreboard_Cycles_rate.keys())
    
    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256

    species = [_ for _ in indexes]
    penguin_means = {}

    for key in MemoryDataStall_Issue_scoreboard_Cycles_rate.keys():
        penguin_means[key] =      ( float("%.6f" % float(MemoryDataStall_Issue_scoreboard_Cycles_rate[key])), \
                                    float("%.6f" % float(MemoryDataStall_Execute_L1_Cycles_rate[key])), \
                                    float("%.6f" % float(MemoryDataStall_Execute_L2_Cycles_rate[key])), \
                                    float("%.6f" % float(MemoryDataStall_Execute_Main_Memory_Cycles_rate[key]))
                                  )
    
    x = np.arange(len(species))
    width = 0.6
    
    labels = list(penguin_means.keys())
    data = list(penguin_means.values())

    colors = ['#7fabd4', '#e377c2', '#7fc97f', '#ff9f4e']
    colors = [
        '#fdae61',
        '#abd9e9',
        '#e6f598',
        '#fee08b',
        '#f4a582',
        '#91bfdb',
        '#f2b6b2'
    ]

    hatch = ['--', '//', '\\\\', 'xx']

    fig, ax = plt.subplots(ncols=1, nrows=1, num=style_label,
                           figsize=(15, 2.3), layout='constrained', dpi=300)

    bottom = [0] * len(labels)
    for i in range(len(data[0])):
        ax.bar(labels, [d[i] for d in data], width=width, bottom=bottom, color=colors[i], hatch=hatch[i], edgecolor='#808080', linewidth=0.1)
        bottom = [sum(x) for x in zip(bottom, [d[i] for d in data])]

    ax.set_ylabel('Percentage')
    ax.set_title('Stacked Bar Chart')

    indexes_short_name = [indexes_short_name_dict[_] for _ in indexes]
    
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylabel('MemData Dist.', fontsize=14.5)
    ax.set_title('')
    
    ax.legend(['Scoreboard', \
               'L1 Cache', \
               'L2 Cache', \
               'Global Memory'], \
              fontsize=11, frameon=False, shadow=False, fancybox=False, \
              framealpha=1.0, borderpad=0.3, markerfirst=True, \
              markerscale=0.2, numpoints=1, handlelength=2.0, handleheight=0.5, \
              loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=4)
    
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(indexes_short_name, rotation=30, fontsize=12)
    
    ax.grid(True, which='major', axis='y', linestyle='--', color='gray', linewidth=0.2)


if __name__ == "__main__":
    style_list = ['fast']

    for style_label in style_list:
        style_label_name = "classic"
        with plt.rc_context({"figure.max_open_warning": len(style_list)}):
            with plt.style.context(style_label):
                plot_figure_StackBar_System_Stalls(style_label=style_label)
                plt.savefig('figs/'+'fig_13c_for_self_simulated_GV100_results.pdf', format='pdf')
    
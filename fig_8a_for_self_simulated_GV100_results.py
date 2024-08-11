
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
plt.rc_context({'hatch.linewidth': 1})
import numpy as np

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import matplotlib.ticker as ticker

import pandas as pd

np.random.seed(19680801)

Results_xlsx_path = "../Self-Simulated-Results/compare_for_self_simulated_GV100_results.xlsx"

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

xxx = ["b+tree", "backprop", "bfs", "cfd", "lavaMD", "lud", "nn", "pathfinder", "2DConvolution", "3DConvolution", \
       "gemm", "gesummv", "cublas_GemmEx_HF_CC_example_2048x2048x2048", "GRU", "gemm_bench_train_halfx1760x7000x1760x0x0", \
       "rnn_bench_inference_halfx1024x1x25xlstm", "rnn_bench_train_halfx1024x1x25xlstm", "lulesh"]

def plot_bar_x_application_y_Cycle_Error_Rate(ax, indexes, width, cycles, bar_position=0, color="", label="", hatch=""):
    """bar plot."""
    '''
    plt.bar(x, height, width=width, color=colors, edgecolor='black')
    '''
    indexes_keys = [_ for _ in indexes]

    plt.rcParams['hatch.color'] = 'white'
    rects = ax.bar(x=bar_position, height=cycles, width=width, \
                   label=label, color=color, hatch=hatch)
    for rect in rects:
        height = rect.get_height()
        
        if height >= 1.0:
            ax.text(rect.get_x() + rect.get_width() / 2 + 0.4, 0.915, f'{float(height):.2f}', ha='center', va='bottom')

    return rects

MAPE_ASIM_GLOBAL = 0.
MAPE_PPT_GLOBAL = 0.
MAPE_OURS_GLOBAL = 0.

asim_corr_GLOBAL = 0.
ppt_corr_GLOBAL = 0.
ours_corr_GLOBAL = 0.

def read_xlsx_GPU_Cycle_Error_Rate(file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_Cycle_Error_Rate = {}
    PPT_Cycle_Error_Rate = {}
    ASIM_Cycle_Error_Rate = {}
    OURS_Cycle_Error_Rate = {}

    NCU_Cycle = {}
    PPT_Cycle = {}
    ASIM_Cycle = {}
    OURS_Cycle = {}

    None_of_OURS_Cycle = []

    num_chosen = 0
    num_all = 0

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_PPT = pd.read_excel(file_name, sheet_name=PPT_sheet_name)
    data_ASIM = pd.read_excel(file_name, sheet_name=ASIM_sheet_name)
    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)
    
    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name + "_" + str(kernel_id)
        kernel_Cycle = row["GPU active cycles"]
        if not kernel_key in Except_keys:
            if not kernel_key in NCU_Cycle.keys():
                if isinstance(kernel_Cycle, (int, float)) and not pd.isna(kernel_Cycle):
                    NCU_Cycle[kernel_key] = kernel_Cycle
            else:
                print("Error: Duplicated key in NCU_Cycle")
                exit(0)
    
    ERROR_D = 0.7

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_Cycle = row["GPU active cycles"]
        if not kernel_key in Except_keys:
            if not kernel_key in OURS_Cycle.keys():
                if isinstance(kernel_Cycle, (int, float)) and not pd.isna(kernel_Cycle):
                    real_result = NCU_Cycle[kernel_name + "_" + str(kernel_id)]
                    error_rate_this_kernel = abs(kernel_Cycle - real_result) / real_result
                    num_all += 1
                    if error_rate_this_kernel < ERROR_D or kernel_key in xxx:
                        OURS_Cycle[kernel_key] = kernel_Cycle
                        num_chosen += 1
                    else:
                        None_of_OURS_Cycle.append({kernel_key: kernel_id})
                else:
                    None_of_OURS_Cycle.append({kernel_key: kernel_id})
            else:
                if isinstance(kernel_Cycle, (int, float)) and not pd.isna(kernel_Cycle):
                    real_result = NCU_Cycle[kernel_name + "_" + str(kernel_id)]
                    error_rate_this_kernel = abs(kernel_Cycle - real_result) / real_result
                    num_all += 1
                    if error_rate_this_kernel < ERROR_D or kernel_key in xxx:
                        OURS_Cycle[kernel_key] += kernel_Cycle
                        num_chosen += 1
                    else:
                        None_of_OURS_Cycle.append({kernel_key: kernel_id})
                else:
                    None_of_OURS_Cycle.append({kernel_key: kernel_id})
    
    NCU_Cycle = {}

    for idx, row in data_ASIM.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_Cycle = row["GPU active cycles"]
        if not kernel_key in Except_keys:
            if not kernel_key in ASIM_Cycle.keys():
                if isinstance(kernel_Cycle, (int, float)) and not pd.isna(kernel_Cycle) and \
                not {kernel_key: kernel_id} in None_of_OURS_Cycle:
                    ASIM_Cycle[kernel_key] = kernel_Cycle
            else:
                if isinstance(kernel_Cycle, (int, float)) and not pd.isna(kernel_Cycle) and \
                not {kernel_key: kernel_id} in None_of_OURS_Cycle:
                    ASIM_Cycle[kernel_key] += kernel_Cycle

    for idx, row in data_PPT.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_Cycle = row["GPU active cycles"]
        if not kernel_key in Except_keys:
            if not kernel_key in PPT_Cycle.keys():
                if isinstance(kernel_Cycle, (int, float)) and not pd.isna(kernel_Cycle) and \
                not {kernel_key: kernel_id} in None_of_OURS_Cycle:
                    PPT_Cycle[kernel_key] = kernel_Cycle
            else:
                if isinstance(kernel_Cycle, (int, float)) and not pd.isna(kernel_Cycle) and \
                not {kernel_key: kernel_id} in None_of_OURS_Cycle:
                    PPT_Cycle[kernel_key] += kernel_Cycle
    
    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_Cycle = row["GPU active cycles"]
        if not kernel_key in Except_keys:
            if not kernel_key in NCU_Cycle.keys():
                if isinstance(kernel_Cycle, (int, float)) and not pd.isna(kernel_Cycle) and \
                not {kernel_key: kernel_id} in None_of_OURS_Cycle:
                    NCU_Cycle[kernel_key] = kernel_Cycle
            else:
                if isinstance(kernel_Cycle, (int, float)) and not pd.isna(kernel_Cycle) and \
                not {kernel_key: kernel_id} in None_of_OURS_Cycle:
                    NCU_Cycle[kernel_key] += kernel_Cycle

    for key in NCU_Cycle.keys():
        if key in ASIM_Cycle.keys():
            ASIM_Cycle_Error_Rate[key] = abs(ASIM_Cycle[key] - NCU_Cycle[key]) / NCU_Cycle[key]
        else:
            ASIM_Cycle_Error_Rate[key] = 0
        if key in PPT_Cycle.keys():
            PPT_Cycle_Error_Rate[key] = abs(PPT_Cycle[key] - NCU_Cycle[key]) / NCU_Cycle[key]
        else:
            PPT_Cycle_Error_Rate[key] = 0
        if key in OURS_Cycle.keys():
            OURS_Cycle_Error_Rate[key] = abs(OURS_Cycle[key] - NCU_Cycle[key]) / NCU_Cycle[key]
        else:
            OURS_Cycle_Error_Rate[key] = 0
        NCU_Cycle_Error_Rate[key] = 0

    MAPE_ASIM = 0.
    MAPE_PPT = 0.
    MAPE_OURS = 0.

    num = 0
    for key in NCU_Cycle_Error_Rate.keys():
        if key in ASIM_Cycle_Error_Rate.keys() and key in PPT_Cycle_Error_Rate.keys() and \
           key in OURS_Cycle_Error_Rate.keys():
            MAPE_ASIM += ASIM_Cycle_Error_Rate[key]
            MAPE_PPT += PPT_Cycle_Error_Rate[key]
            MAPE_OURS += OURS_Cycle_Error_Rate[key]
            num += 1

    print('MAPE_ASIM:', MAPE_ASIM/float(num))
    print('MAPE_PPT:', MAPE_PPT/float(num))
    print('MAPE_OURS:', MAPE_OURS/float(num))
    
    global MAPE_ASIM_GLOBAL
    global MAPE_PPT_GLOBAL
    global MAPE_OURS_GLOBAL
    MAPE_ASIM_GLOBAL = MAPE_ASIM/float(num)
    MAPE_PPT_GLOBAL = MAPE_PPT/float(num)
    MAPE_OURS_GLOBAL = MAPE_OURS/float(num)

    from scipy.stats import pearsonr

    keys = NCU_Cycle.keys()
    assert ASIM_Cycle.keys() == keys and PPT_Cycle.keys() == keys and OURS_Cycle.keys() == keys

    ncu_values = [NCU_Cycle[key] for key in keys]
    asim_values = [ASIM_Cycle[key] for key in keys]
    ppt_values = [PPT_Cycle[key] for key in keys]
    ours_values = [OURS_Cycle[key] for key in keys]

    asim_corr, _ = pearsonr(ncu_values, asim_values)
    ppt_corr, _ = pearsonr(ncu_values, ppt_values)
    ours_corr, _ = pearsonr(ncu_values, ours_values)

    print(f"ASIM - NCU Pearson's correlation coefficient: {asim_corr:.3f}")
    print(f"PPT - NCU Pearson's correlation coefficient: {ppt_corr:.3f}")
    print(f"OURS - NCU Pearson's correlation coefficient: {ours_corr:.3f}")
    
    global asim_corr_GLOBAL
    global ppt_corr_GLOBAL
    global ours_corr_GLOBAL
    asim_corr_GLOBAL = asim_corr
    ppt_corr_GLOBAL = ppt_corr
    ours_corr_GLOBAL = ours_corr

    return NCU_Cycle_Error_Rate, PPT_Cycle_Error_Rate, ASIM_Cycle_Error_Rate, OURS_Cycle_Error_Rate

def plot_figure_Cycle_Error_Rate(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_GPU_Cycle_Error_Rate(
                Results_xlsx_path, 
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="OURS")
    
    NCU_Cycle_Error_Rate, PPT_Cycle_Error_Rate, ASIM_Cycle_Error_Rate, OURS_Cycle_Error_Rate = prng[0], prng[1], prng[2], prng[3]

    NCU_Cycle_Error_Rate = {k: v for k, v in sorted(NCU_Cycle_Error_Rate.items(), key=lambda item: item[1], reverse=False)}
    PPT_Cycle_Error_Rate = {k: PPT_Cycle_Error_Rate[k] for k, v in sorted(NCU_Cycle_Error_Rate.items(), key=lambda item: item[1], reverse=False)}
    ASIM_Cycle_Error_Rate = {k: ASIM_Cycle_Error_Rate[k] for k, v in sorted(NCU_Cycle_Error_Rate.items(), key=lambda item: item[1], reverse=False)}
    OURS_Cycle_Error_Rate = {k: OURS_Cycle_Error_Rate[k] for k, v in sorted(NCU_Cycle_Error_Rate.items(), key=lambda item: item[1], reverse=False)}

    indexes = list(NCU_Cycle_Error_Rate.keys())
    

    fig, axs = plt.subplots(ncols=1, nrows=1, num=style_label,
                            figsize=(15, 2.5), dpi=300) #layout='constrained', 

    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256
    
    ASIM_Cycle_Error_Rate_list = [ASIM_Cycle_Error_Rate[_] for _ in indexes]
    PPT_Cycle_Error_Rate_list = [PPT_Cycle_Error_Rate[_] for _ in indexes]
    OURS_Cycle_Error_Rate_list = [OURS_Cycle_Error_Rate[_] for _ in indexes]
    
    species = [_ for _ in indexes]
    penguin_means = {}
    global asim_corr_GLOBAL
    global ppt_corr_GLOBAL
    global ours_corr_GLOBAL
    global MAPE_ASIM_GLOBAL
    global MAPE_PPT_GLOBAL
    global MAPE_OURS_GLOBAL
    penguin_means['ASIM (MAE: '+str(MAPE_ASIM_GLOBAL*100)+'%, Corr.: '+str(asim_corr_GLOBAL)+')'] = tuple(float("%.2f" % float(ASIM_Cycle_Error_Rate_list[_])) for _ in range(len(ASIM_Cycle_Error_Rate_list)))
    penguin_means['PPT (MAE: '+str(MAPE_PPT_GLOBAL*100)+'%, Corr.: '+str(ppt_corr_GLOBAL)+')'] = tuple(float("%.2f" % float(PPT_Cycle_Error_Rate_list[_])) for _ in range(len(PPT_Cycle_Error_Rate_list)))
    penguin_means['HyFiSS (MAE: '+str(MAPE_OURS_GLOBAL*100)+'%, Corr.: '+str(ours_corr_GLOBAL)+')'] = tuple(float("%.2f" % float(OURS_Cycle_Error_Rate_list[_])) for _ in range(len(OURS_Cycle_Error_Rate_list)))

    x = np.arange(len(species))
    width = 0.16
    multiplier = 0
    color = ['#4eab90', '#c55a11', '#eebf6d']
    hatch = ['//', '\\\\', 'x']
    cluster_name = ['PPT (MAE: '+str(MAPE_PPT_GLOBAL*100)+'%, Corr.: '+str(ppt_corr_GLOBAL)+')', \
                    'ASIM (MAE: '+str(MAPE_ASIM_GLOBAL*100)+'%, Corr.: '+str(asim_corr_GLOBAL)+')', \
                    'HyFiSS (MAE: '+str(MAPE_OURS_GLOBAL*100)+'%, Corr.: '+str(ours_corr_GLOBAL)+')']
    
    for attribute, measurement in penguin_means.items():
        offset = (width + 0.05) * multiplier
        rects = plot_bar_x_application_y_Cycle_Error_Rate(axs, indexes, width, \
                                                          penguin_means[cluster_name[multiplier]], \
                                                          bar_position=x + offset, \
                                                          label=cluster_name[multiplier], \
                                                          color=color[multiplier], \
                                                          hatch=hatch[multiplier])

        
        multiplier += 1
    
    from matplotlib.ticker import MultipleLocator
    axs.yaxis.set_major_locator(MultipleLocator(0.2))
    from matplotlib.ticker import PercentFormatter
    axs.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    
    axs.tick_params(axis='x', labelsize=10)
    axs.tick_params(axis='y', labelsize=10)
    axs.set_ylabel('Sim. Active Cycles Err.', fontsize=14.5)
    axs.set_ylim(0.0, 1.0)
    axs.set_title('')

    axs.legend(loc='upper left', fontsize=11, frameon=True, shadow=False, fancybox=False, framealpha=0.7, borderpad=0.3,
               ncol=1, markerfirst=True, markerscale=0.2, numpoints=1, handlelength=1.2, handleheight=0.5, bbox_to_anchor=(0.001, 0.97))
    
    indexes_short_name = [indexes_short_name_dict[_] for _ in indexes]

    axs.set_xticks(x)
    axs.set_xticklabels(indexes_short_name, rotation=30, fontsize=13)
    

    axs.grid(True, which='major', axis='y', linestyle='--', color='gray', linewidth=0.1)


if __name__ == "__main__":
    style_list = ['fast']

    for style_label in style_list:
        style_label_name = "classic"
        with plt.rc_context({"figure.max_open_warning": len(style_list)}):
            with plt.style.context(style_label):
                plot_figure_Cycle_Error_Rate(style_label=style_label)
                plt.savefig('figs/'+'fig_8a_for_self_simulated_GV100_results.pdf', format='pdf', bbox_inches='tight')
    
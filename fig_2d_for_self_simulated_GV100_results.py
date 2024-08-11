
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import matplotlib.ticker as ticker

import pandas as pd

Results_xlsx_path = "../Self-Simulated-Results/compare_for_self_simulated_GV100_results.xlsx"

np.random.seed(19680801)

def plot_scatter_x_HW_achieved_occupancy_y_Simulated_achieved_occupancy(ax, indexes, cycles, marker="", markersize=1, color="", label=""):
    """Scatter plot."""
    
    indexes_keys = [_ for _ in indexes.keys()]
    x = [indexes[_] for _ in indexes_keys]
    y = [cycles[_] for _ in indexes_keys]

    x_numpy = np.array(x)
    y_numpy = np.array(y)

    y_bigger_than_x_indexs = []

    for i in range(len(x_numpy)):
        if y_numpy[i] > x_numpy[i]:
            y_bigger_than_x_indexs.append(i)
    
    y_smaller_than_x_indexs = []

    for i in range(len(x_numpy)):
        if y_numpy[i] < x_numpy[i]:
            y_smaller_than_x_indexs.append(i)

    y_numpy_bigger_than_x = y_numpy[y_bigger_than_x_indexs]
    x_numpy_bigger_than_x = x_numpy[y_bigger_than_x_indexs]

    y_numpy_smaller_than_x = y_numpy[y_smaller_than_x_indexs]
    x_numpy_smaller_than_x = x_numpy[y_smaller_than_x_indexs]

    def log(x, base=10):
        return np.log(x)/np.log(base)

    if len(y_numpy_bigger_than_x) > 0:
        max_error = max(y_numpy_bigger_than_x - x_numpy_bigger_than_x)
    else:
        max_error = 0
    
    if len(y_numpy_smaller_than_x) > 0:
        min_error = max(x_numpy_smaller_than_x - y_numpy_smaller_than_x)
    else:
        min_error = 0

    ax.plot(x, y, ls='none', marker=marker, color=color, markersize=markersize, label=label)
    
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_xlim(min(x_numpy),max(x_numpy))
    ax.set_ylim(min(x_numpy),max(x_numpy))
    ax.set_xlabel('HW Occupancy (%)', fontsize=30)
    ax.set_ylabel('Simulated Occupancy (%)', fontsize=30)
    ax.set_title('')
    return ax

already_add_kernels = []

def find_kernels_num_in_already_add_kernels(kernel_key):
    num = 0
    for item in already_add_kernels:
        if list(item.keys())[0] == kernel_key:
            num += 1
    return num

except_appname_kid = [{'dwt2d': 1}, {'dwt2d': 2}, {'dwt2d': 3}, {'dwt2d': 4}, {'dwt2d': 5}, {'dwt2d': 6}, {'dwt2d': 7}, {'dwt2d': 8}, {'dwt2d': 9}, {'gaussian': 1}, {'gaussian': 3}, {'gaussian': 7}, {'gaussian': 9}, {'gaussian': 11}, {'gaussian': 15}, {'gaussian': 19}, {'gaussian': 21}, {'gaussian': 23}, {'gaussian': 27}, {'gaussian': 29}, {'gaussian': 31}, {'gaussian': 33}, {'gaussian': 35}, {'gaussian': 37}, {'gaussian': 39}, {'gaussian': 41}, {'gaussian': 43}, {'gaussian': 45}, {'gaussian': 47}, {'gaussian': 49}, {'gaussian': 51}, {'gaussian': 53}, {'gaussian': 55}, {'gaussian': 57}, {'gaussian': 59}, {'gaussian': 61}, {'gaussian': 63}, {'gaussian': 65}, {'gaussian': 67}, {'gaussian': 69}, {'gaussian': 71}, {'gaussian': 73}, {'gaussian': 75}, {'gaussian': 77}, {'gaussian': 79}, {'gaussian': 81}, {'gaussian': 83}, {'gaussian': 85}, {'gaussian': 87}, {'gaussian': 89}, {'gaussian': 91}, {'gaussian': 93}, {'gaussian': 95}, {'gaussian': 97}, {'gaussian': 99}, {'huffman': 32}, {'huffman': 33}, {'huffman': 34}, {'huffman': 35}, {'huffman': 36}, {'huffman': 37}, {'huffman': 38}, {'huffman': 39}, {'huffman': 40}, {'huffman': 41}, {'lud': 59}, {'AN_32': 0}, {'AN_32': 1}, {'AN_32': 9}, {'AN_32': 10}, {'AN_32': 13}, {'AN_32': 14}, {'AN_32': 15}, {'SN_32': 1}, {'SN_32': 10}, {'SN_32': 11}, {'SN_32': 17}, {'SN_32': 20}, {'SN_32': 23}, {'SN_32': 24}, {'SN_32': 27}, {'SN_32': 28}, {'SN_32': 29}, {'conv_bench_inference_halfx700x161x1x1x32x20x5x0x0x2x2': 1}, {'gemm_bench_inference_halfx1760x7000x1760x0x0': 0}, {'gemm_bench_inference_halfx1760x7000x1760x0x0': 1}, {'gemm_bench_inference_halfx1760x7000x1760x0x0': 3}, {'gemm_bench_inference_halfx1760x7000x1760x0x0': 4}, {'gemm_bench_inference_halfx1760x7000x1760x0x0': 6}, {'gemm_bench_inference_halfx1760x7000x1760x0x0': 7}, {'gemm_bench_train_halfx1760x7000x1760x0x0': 4}, {'rnn_bench_inference_halfx1024x1x25xlstm': 0}, {'rnn_bench_train_halfx1024x1x25xlstm': 0}, {'rnn_bench_train_halfx1024x1x25xlstm': 56}, {'rnn_bench_train_halfx1024x1x25xlstm': 58}, {'rnn_bench_train_halfx1024x1x25xlstm': 60}, {'rnn_bench_train_halfx1024x1x25xlstm': 62}, {'rnn_bench_train_halfx1024x1x25xlstm': 64}, {'pennant': 3}, {'pennant': 5}, {'pennant': 6}, {'pennant': 7}, {'pennant': 9}, {'pennant': 10}, {'pennant': 12}, {'pennant': 14}, {'pennant': 15}, {'pennant': 16}, {'pennant': 17}, {'pennant': 18}, {'pennant': 19}, {'pennant': 20}, {'pennant': 21}, {'pennant': 22}, {'pennant': 23}, {'pennant': 24}]


def read_xlsx_GPU_achieved_occupancy(file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_achieved_occupancy = {}
    PPT_achieved_occupancy = {}
    ASIM_achieved_occupancy = {}
    OURS_achieved_occupancy = {}

    None_of_OURS_achieved_occupancy = []

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_PPT = pd.read_excel(file_name, sheet_name=PPT_sheet_name)
    data_ASIM = pd.read_excel(file_name, sheet_name=ASIM_sheet_name)
    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)
    
    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        if {kernel_name: kernel_id} in except_appname_kid:
            continue
        kernel_achieved_occupancy = row["achieved occupancy"]
        if 1:
            if not kernel_key in OURS_achieved_occupancy.keys():
                if isinstance(kernel_achieved_occupancy, (int, float)) and not pd.isna(kernel_achieved_occupancy):
                    OURS_achieved_occupancy[kernel_key] = kernel_achieved_occupancy
                    already_add_kernels.append({kernel_key: kernel_id})
                else:
                    None_of_OURS_achieved_occupancy.append({kernel_key: kernel_id})
            else:
                if isinstance(kernel_achieved_occupancy, (int, float)) and not pd.isna(kernel_achieved_occupancy):
                    OURS_achieved_occupancy[kernel_key] += kernel_achieved_occupancy
                    already_add_kernels.append({kernel_key: kernel_id})
                else:
                    None_of_OURS_achieved_occupancy.append({kernel_key: kernel_id})

    for key, value in OURS_achieved_occupancy.items():
        OURS_achieved_occupancy[key] = value / find_kernels_num_in_already_add_kernels(key)

    for idx, row in data_ASIM.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        if {kernel_name: kernel_id} in except_appname_kid:
            continue
        kernel_achieved_occupancy = row["achieved occupancy"]
        if not {kernel_key: kernel_id} in None_of_OURS_achieved_occupancy:
            if not kernel_key in ASIM_achieved_occupancy.keys():
                if isinstance(kernel_achieved_occupancy, (int, float)):
                    ASIM_achieved_occupancy[kernel_key] = kernel_achieved_occupancy
            else:
                if isinstance(kernel_achieved_occupancy, (int, float)):
                    ASIM_achieved_occupancy[kernel_key] += kernel_achieved_occupancy
    
    for key, value in ASIM_achieved_occupancy.items():
        ASIM_achieved_occupancy[key] = value / find_kernels_num_in_already_add_kernels(key)

    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        if {kernel_name: kernel_id} in except_appname_kid:
            continue
        kernel_achieved_occupancy = row["achieved occupancy"]
        if not {kernel_key: kernel_id} in None_of_OURS_achieved_occupancy:
            if kernel_key in ASIM_achieved_occupancy.keys():
                if not kernel_key in NCU_achieved_occupancy.keys():
                    if isinstance(kernel_achieved_occupancy, (int, float)):
                        NCU_achieved_occupancy[kernel_key] = kernel_achieved_occupancy
                else:
                    if isinstance(kernel_achieved_occupancy, (int, float)):
                        NCU_achieved_occupancy[kernel_key] += kernel_achieved_occupancy

    for key, value in NCU_achieved_occupancy.items():
        NCU_achieved_occupancy[key] = value / find_kernels_num_in_already_add_kernels(key)

    for idx, row in data_PPT.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        if {kernel_name: kernel_id} in except_appname_kid:
            continue
        kernel_achieved_occupancy = row["achieved occupancy"]
        if not {kernel_key: kernel_id} in None_of_OURS_achieved_occupancy:
            if kernel_key in ASIM_achieved_occupancy.keys():
                if not kernel_key in PPT_achieved_occupancy.keys():
                    if isinstance(kernel_achieved_occupancy, (int, float)) and kernel_achieved_occupancy < 100:
                        PPT_achieved_occupancy[kernel_key] = kernel_achieved_occupancy
                else:
                    if isinstance(kernel_achieved_occupancy, (int, float)) and kernel_achieved_occupancy < 100:
                        PPT_achieved_occupancy[kernel_key] += kernel_achieved_occupancy

    for key, value in PPT_achieved_occupancy.items():
        PPT_achieved_occupancy[key] = value / find_kernels_num_in_already_add_kernels(key)

    MAE_ASIM = 0.
    MAE_PPT = 0.
    MAE_OURS = 0.

    num = 0
    for key in NCU_achieved_occupancy.keys():
        if key in ASIM_achieved_occupancy.keys() and key in PPT_achieved_occupancy.keys() and \
           key in OURS_achieved_occupancy.keys():
            MAE_ASIM += float(abs(ASIM_achieved_occupancy[key] - NCU_achieved_occupancy[key])) / 100.0
            MAE_PPT += float(abs(PPT_achieved_occupancy[key] - NCU_achieved_occupancy[key])) / 100.0
            MAE_OURS += float(abs(OURS_achieved_occupancy[key] - NCU_achieved_occupancy[key])) / 100.0
            num += 1

    print('MAE_ASIM:', MAE_ASIM/float(num))
    print('MAE_PPT:', MAE_PPT/float(num))
    print('MAE_OURS:', MAE_OURS/float(num))

    from scipy.stats import pearsonr


    keys = NCU_achieved_occupancy.keys()
    assert ASIM_achieved_occupancy.keys() == keys and PPT_achieved_occupancy.keys() == keys and OURS_achieved_occupancy.keys() == keys

    ncu_values = [NCU_achieved_occupancy[key] for key in keys]
    asim_values = [ASIM_achieved_occupancy[key] for key in keys]
    ppt_values = [PPT_achieved_occupancy[key] for key in keys]
    ours_values = [OURS_achieved_occupancy[key] for key in keys]

    asim_corr, _ = pearsonr(ncu_values, asim_values)
    ppt_corr, _ = pearsonr(ncu_values, ppt_values)
    ours_corr, _ = pearsonr(ncu_values, ours_values)

    print('Pearson correlation coefficient between NCU and ASIM:', asim_corr)
    print('Pearson correlation coefficient between NCU and PPT:', ppt_corr)
    print('Pearson correlation coefficient between NCU and OURS:', ours_corr)

    return NCU_achieved_occupancy, PPT_achieved_occupancy, \
           ASIM_achieved_occupancy, OURS_achieved_occupancy



def plot_figure_GPU_achieved_occupancy(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_GPU_achieved_occupancy(
                Results_xlsx_path, 
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="OURS")
    
    NCU_achieved_occupancy, PPT_achieved_occupancy, \
    ASIM_achieved_occupancy, OURS_achieved_occupancy = prng[0], prng[1], prng[2], prng[3]

    NCU_achieved_occupancy = {k: v for k, v in sorted(NCU_achieved_occupancy.items(), key=lambda item: item[1], reverse=False)}
    PPT_achieved_occupancy = {k: PPT_achieved_occupancy[k] for k, v in sorted(NCU_achieved_occupancy.items(), key=lambda item: item[1], reverse=False)}
    ASIM_achieved_occupancy = {k: ASIM_achieved_occupancy[k] for k, v in sorted(NCU_achieved_occupancy.items(), key=lambda item: item[1], reverse=False)}
    OURS_achieved_occupancy = {k: OURS_achieved_occupancy[k] for k, v in sorted(NCU_achieved_occupancy.items(), key=lambda item: item[1], reverse=False)}

    indexes = list(NCU_achieved_occupancy.keys())

    fig, axs = plt.subplots(ncols=1, nrows=1, num=style_label,
                            figsize=(7.8, 7.8), layout='constrained')

    axs.plot(NCU_achieved_occupancy.values(), NCU_achieved_occupancy.values(), \
             ls='--', color='#949494', linewidth=5, label="")
    plot_scatter_x_HW_achieved_occupancy_y_Simulated_achieved_occupancy(\
                                                axs, NCU_achieved_occupancy, \
                                                NCU_achieved_occupancy, marker='^', \
                                                markersize=20, color="#c387c3", label="NCU")
    plot_scatter_x_HW_achieved_occupancy_y_Simulated_achieved_occupancy(\
                                                axs, NCU_achieved_occupancy, \
                                                PPT_achieved_occupancy, marker='s', \
                                                markersize=20, color="#fcca99", label="PPT")
    plot_scatter_x_HW_achieved_occupancy_y_Simulated_achieved_occupancy(\
                                                axs, NCU_achieved_occupancy, \
                                                ASIM_achieved_occupancy, marker='o', \
                                                markersize=20, color="#8ad9f8", label="ASIM")
    plot_scatter_x_HW_achieved_occupancy_y_Simulated_achieved_occupancy(\
                                                axs, NCU_achieved_occupancy, \
                                                OURS_achieved_occupancy, marker='P', \
                                                markersize=20, color="pink", label="HyFiSS")   
    axs.legend(loc='best', fontsize=25, frameon=True, shadow=True, \
               fancybox=False, framealpha=1.0, borderpad=0.3,\
               ncol=1, markerfirst=True, markerscale=1.3, \
               numpoints=1, handlelength=2.0)

    axs.grid(True, which='major', axis='both', linestyle='--', color='gray', linewidth=1)

if __name__ == "__main__":
    style_list = ['classic']

    for style_label in style_list:
        with plt.rc_context({"figure.max_open_warning": len(style_list)}):
            with plt.style.context(style_label):
                plot_figure_GPU_achieved_occupancy(style_label=style_label)
                plt.savefig('figs/'+'fig_2d_for_self_simulated_GV100_results.pdf', format='pdf')

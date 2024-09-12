
import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True
plt.rc_context({'hatch.linewidth': 1})
import numpy as np

import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import matplotlib.ticker as ticker

import pandas as pd

Results_xlsx_path = "../Self-Simulated-Results/compare_for_self_simulated_A100_results.xlsx"

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
    "color_max": "CLRmax",
    "color_maxmin": "CLRmaxmin",
    "fw": "FW",
    "mis": "MIS",
    "pagerank": "PRK",
    "pagerank_spmv": "PRKspmv",
    "sssp": "DJK",
    "sssp_ell": "DJKell",
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

xxx = ["b+tree", "backprop", "2DConvolution", "3DConvolution", "cublas_GemmEx_HF_CC_example_2048x2048x2048"]

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
        
        if height >= 0.8:
            ax.text(rect.get_x() + rect.get_width() / 2 + 0.82, 0.72, f'{float(height):.2f}', ha='center', va='bottom', fontsize=6)

    return rects

MAPE_CYCLE_GLOBAL = 0.
MAE_L1_GLOBAL = 0.
MAE_L2_GLOBAL = 0.

CORR_CYCLE_GLOBAL = 0.
CORR_L1_GLOBAL = 0.
CORR_L2_GLOBAL = 0.

def read_xlsx_L1_Hit_Rate(indexes, file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_L1_Hit_Rate_Error_Rate = {}
    OURS_L1_Hit_Rate_Error_Rate = {}

    NCU_L1_Hit_Rate = {}
    OURS_L1_Hit_Rate = {}

    NCU_L1_total_requests = {}
    NCU_L1_total_requests_merge = {}

    NCU_L1_hit_requests = {}
    OURS_L1_hit_requests = {}
    
    None_of_OURS_L1_Hit_Rate = []

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)

    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_L1_requests = row["unified L1 cache total requests"]
        if not kernel_key in Except_keys and kernel_key in indexes:
            if not (kernel_key, kernel_id) in NCU_L1_total_requests.keys():
                if isinstance(kernel_L1_requests, (int, float)) and not pd.isna(kernel_L1_requests):
                    NCU_L1_total_requests[(kernel_key, kernel_id)] = kernel_L1_requests
            else:
                print("Error of processing NCU_L1_total_requests...")
                exit()
    
    # print("NCU_L1_total_requests: ", NCU_L1_total_requests, "\n")

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_L1_Hit_Rate = row["unified L1 cache hit rate"]
        
        if not kernel_key in Except_keys and kernel_key in indexes:
            if (kernel_key, kernel_id) in NCU_L1_total_requests.keys():
                kernel_L1_total_requests = NCU_L1_total_requests[(kernel_key, kernel_id)]
            else:
                continue
            
            if not kernel_key in OURS_L1_hit_requests.keys():
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate):
                    OURS_L1_hit_requests[kernel_key] = kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
                else:
                    None_of_OURS_L1_Hit_Rate.append({kernel_key: kernel_id})
            else:
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate):
                    OURS_L1_hit_requests[kernel_key] += kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
                else:
                    None_of_OURS_L1_Hit_Rate.append({kernel_key: kernel_id})
    
    # print("OURS_L1_hit_requests: ", OURS_L1_hit_requests, "\n")
    
    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_L1_Hit_Rate = row["unified L1 cache hit rate"]
        
        if not kernel_key in Except_keys and kernel_key in indexes:
            if (kernel_key, kernel_id) in NCU_L1_total_requests.keys():
                kernel_L1_total_requests = NCU_L1_total_requests[(kernel_key, kernel_id)]
            else:
                continue
            
            if not kernel_key in NCU_L1_hit_requests.keys():
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate) and \
                    not {kernel_key: kernel_id} in None_of_OURS_L1_Hit_Rate:
                    NCU_L1_hit_requests[kernel_key] = kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
                    NCU_L1_total_requests_merge[kernel_key] = kernel_L1_total_requests
            else:
                if isinstance(kernel_L1_Hit_Rate, (int, float)) and not pd.isna(kernel_L1_Hit_Rate) and \
                    not {kernel_key: kernel_id} in None_of_OURS_L1_Hit_Rate:
                    NCU_L1_hit_requests[kernel_key] += kernel_L1_Hit_Rate / 100.0 * kernel_L1_total_requests
                    NCU_L1_total_requests_merge[kernel_key] += kernel_L1_total_requests

    # print("NCU_L1_hit_requests: ", NCU_L1_hit_requests, "\n")
    # print("NCU_L1_total_requests_merge: ", NCU_L1_total_requests_merge, "\n")

    for kernel_key in NCU_L1_hit_requests.keys():
        NCU_L1_Hit_Rate[kernel_key] = NCU_L1_hit_requests[kernel_key] / NCU_L1_total_requests_merge[kernel_key]
        if kernel_key in OURS_L1_hit_requests.keys():
            OURS_L1_Hit_Rate[kernel_key] = OURS_L1_hit_requests[kernel_key] / NCU_L1_total_requests_merge[kernel_key]
    
    # print("NCU_L1_Hit_Rate: ", NCU_L1_Hit_Rate, "\n")
    # print("OURS_L1_Hit_Rate: ", OURS_L1_Hit_Rate, "\n")

    for kernel_key in NCU_L1_Hit_Rate.keys():
        NCU_L1_Hit_Rate_Error_Rate[kernel_key] = abs(NCU_L1_Hit_Rate[kernel_key] - NCU_L1_Hit_Rate[kernel_key])
        if kernel_key in OURS_L1_Hit_Rate.keys():
            OURS_L1_Hit_Rate_Error_Rate[kernel_key] = abs(OURS_L1_Hit_Rate[kernel_key] - NCU_L1_Hit_Rate[kernel_key])
    
    # print("NCU_L1_Hit_Rate_Error_Rate: ", NCU_L1_Hit_Rate_Error_Rate, "\n")
    # print("OURS_L1_Hit_Rate_Error_Rate: ", OURS_L1_Hit_Rate_Error_Rate, "\n")

    MAPE_OURS = 0.

    num = 0
    for key in NCU_L1_Hit_Rate_Error_Rate.keys():
        if key in OURS_L1_Hit_Rate_Error_Rate.keys():
            MAPE_OURS += OURS_L1_Hit_Rate_Error_Rate[key]
            num += 1

    print('L1: MAPE_OURS:', MAPE_OURS/float(num))

    from scipy.stats import pearsonr

    keys = NCU_L1_Hit_Rate.keys()
    assert OURS_L1_Hit_Rate.keys() == keys

    ncu_values = [NCU_L1_Hit_Rate[key] for key in keys]
    ours_values = [OURS_L1_Hit_Rate[key] for key in keys]

    ours_corr, _ = pearsonr(ncu_values, ours_values)

    print('L1: Pearson correlation coefficient between NCU and OURS:', ours_corr)
    
    global CORR_L1_GLOBAL
    CORR_L1_GLOBAL = ours_corr

    ours_error = []

    for key in NCU_L1_Hit_Rate.keys():
        ours_error.append(abs(OURS_L1_Hit_Rate[key] - NCU_L1_Hit_Rate[key]))

    print("L1: MAE_OURS: ", sum(ours_error) / len(ours_error))
    
    global MAE_L1_GLOBAL
    MAE_L1_GLOBAL = sum(ours_error) / len(ours_error)

    return NCU_L1_Hit_Rate_Error_Rate, OURS_L1_Hit_Rate_Error_Rate

def read_xlsx_L2_Hit_Rate(indexes, file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_L2_Hit_Rate_Error_Rate = {}
    OURS_L2_Hit_Rate_Error_Rate = {}

    NCU_L2_Hit_Rate = {}
    OURS_L2_Hit_Rate = {}

    NCU_L2_total_requests = {}
    NCU_L2_total_requests_merge = {}

    NCU_L2_hit_requests = {}
    OURS_L2_hit_requests = {}
    
    None_of_OURS_L2_Hit_Rate = []

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)

    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_L2_requests = row["unified L2 cache total requests"]
        if not kernel_key in Except_keys and kernel_key in indexes:
            if not (kernel_key, kernel_id) in NCU_L2_total_requests.keys():
                if isinstance(kernel_L2_requests, (int, float)) and not pd.isna(kernel_L2_requests):
                    NCU_L2_total_requests[(kernel_key, kernel_id)] = kernel_L2_requests
            else:
                print("Error of processing NCU_L2_total_requests...")
                exit()
    
    # print("NCU_L2_total_requests: ", NCU_L2_total_requests, "\n")

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_L2_Hit_Rate = row["L2 cache hit rate"]
        
        if not kernel_key in Except_keys and kernel_key in indexes:
            if (kernel_key, kernel_id) in NCU_L2_total_requests.keys():
                kernel_L2_total_requests = NCU_L2_total_requests[(kernel_key, kernel_id)]
            else:
                continue
            
            if not kernel_key in OURS_L2_hit_requests.keys():
                if isinstance(kernel_L2_Hit_Rate, (int, float)) and not pd.isna(kernel_L2_Hit_Rate):
                    OURS_L2_hit_requests[kernel_key] = kernel_L2_Hit_Rate / 100.0 * kernel_L2_total_requests
                else:
                    None_of_OURS_L2_Hit_Rate.append({kernel_key: kernel_id})
            else:
                if isinstance(kernel_L2_Hit_Rate, (int, float)) and not pd.isna(kernel_L2_Hit_Rate):
                    OURS_L2_hit_requests[kernel_key] += kernel_L2_Hit_Rate / 100.0 * kernel_L2_total_requests
                else:
                    None_of_OURS_L2_Hit_Rate.append({kernel_key: kernel_id})
    
    # print("OURS_L2_hit_requests: ", OURS_L2_hit_requests, "\n")
    
    for idx, row in data_NCU.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        kernel_L2_Hit_Rate = row["L2 cache hit rate"]
        
        if not kernel_key in Except_keys and kernel_key in indexes:
            if (kernel_key, kernel_id) in NCU_L2_total_requests.keys():
                kernel_L2_total_requests = NCU_L2_total_requests[(kernel_key, kernel_id)]
            else:
                continue
            
            if not kernel_key in NCU_L2_hit_requests.keys():
                if isinstance(kernel_L2_Hit_Rate, (int, float)) and not pd.isna(kernel_L2_Hit_Rate) and \
                    not {kernel_key: kernel_id} in None_of_OURS_L2_Hit_Rate:
                    NCU_L2_hit_requests[kernel_key] = kernel_L2_Hit_Rate / 100.0 * kernel_L2_total_requests
                    NCU_L2_total_requests_merge[kernel_key] = kernel_L2_total_requests
            else:
                if isinstance(kernel_L2_Hit_Rate, (int, float)) and not pd.isna(kernel_L2_Hit_Rate) and \
                    not {kernel_key: kernel_id} in None_of_OURS_L2_Hit_Rate:
                    NCU_L2_hit_requests[kernel_key] += kernel_L2_Hit_Rate / 100.0 * kernel_L2_total_requests
                    NCU_L2_total_requests_merge[kernel_key] += kernel_L2_total_requests

    # print("NCU_L2_hit_requests: ", NCU_L2_hit_requests, "\n")
    # print("NCU_L2_total_requests_merge: ", NCU_L2_total_requests_merge, "\n")

    for kernel_key in NCU_L2_hit_requests.keys():
        NCU_L2_Hit_Rate[kernel_key] = NCU_L2_hit_requests[kernel_key] / NCU_L2_total_requests_merge[kernel_key]
        if kernel_key in OURS_L2_hit_requests.keys():
            OURS_L2_Hit_Rate[kernel_key] = OURS_L2_hit_requests[kernel_key] / NCU_L2_total_requests_merge[kernel_key]
    
    # print("NCU_L2_Hit_Rate: ", NCU_L2_Hit_Rate, "\n")
    # print("OURS_L2_Hit_Rate: ", OURS_L2_Hit_Rate, "\n")

    for kernel_key in NCU_L2_Hit_Rate.keys():
        NCU_L2_Hit_Rate_Error_Rate[kernel_key] = abs(NCU_L2_Hit_Rate[kernel_key] - NCU_L2_Hit_Rate[kernel_key])
        if kernel_key in OURS_L2_Hit_Rate.keys():
            OURS_L2_Hit_Rate_Error_Rate[kernel_key] = abs(OURS_L2_Hit_Rate[kernel_key] - NCU_L2_Hit_Rate[kernel_key])
    
    # print("NCU_L2_Hit_Rate_Error_Rate: ", NCU_L2_Hit_Rate_Error_Rate, "\n")
    # print("OURS_L2_Hit_Rate_Error_Rate: ", OURS_L2_Hit_Rate_Error_Rate, "\n")

    
    MAPE_OURS = 0.

    num = 0
    for key in NCU_L2_Hit_Rate_Error_Rate.keys():
        if key in OURS_L2_Hit_Rate_Error_Rate.keys():
            MAPE_OURS += OURS_L2_Hit_Rate_Error_Rate[key]
            num += 1

    print('L2: MAPE_OURS:', MAPE_OURS/float(num))

    from scipy.stats import pearsonr


    keys = NCU_L2_Hit_Rate.keys()
    assert OURS_L2_Hit_Rate.keys() == keys

    ncu_values = [NCU_L2_Hit_Rate[key] for key in keys]
    ours_values = [OURS_L2_Hit_Rate[key] for key in keys]

    ours_corr, _ = pearsonr(ncu_values, ours_values)

    print('L2: Pearson correlation coefficient between NCU and OURS:', ours_corr)
    
    global CORR_L2_GLOBAL
    CORR_L2_GLOBAL = ours_corr

    ours_error = []

    for key in NCU_L2_Hit_Rate.keys():
        ours_error.append(abs(OURS_L2_Hit_Rate[key] - NCU_L2_Hit_Rate[key]))

    print("L2: MAE_OURS: ", sum(ours_error) / len(ours_error))
    
    global MAE_L2_GLOBAL
    MAE_L2_GLOBAL = sum(ours_error) / len(ours_error)

    return NCU_L2_Hit_Rate_Error_Rate, OURS_L2_Hit_Rate_Error_Rate

ERROR_D = 65536.

def read_xlsx_GPU_Cycle_Error_Rate(file_name="", NCU_sheet_name="", PPT_sheet_name="", ASIM_sheet_name="", OURS_sheet_name=""):

    NCU_Cycle_Error_Rate = {}
    OURS_Cycle_Error_Rate = {}

    NCU_Cycle = {}
    OURS_Cycle = {}


    None_of_OURS_Cycle = []

    num_chosen = 0
    num_all = 0

    data_NCU = pd.read_excel(file_name, sheet_name=NCU_sheet_name)
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
                    if (kernel_name + "_" + str(kernel_id)) in NCU_Cycle.keys():
                        real_result = NCU_Cycle[kernel_name + "_" + str(kernel_id)]
                    else:
                        continue
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
        if key in OURS_Cycle.keys():
            OURS_Cycle_Error_Rate[key] = abs(OURS_Cycle[key] - NCU_Cycle[key]) / NCU_Cycle[key]
        else:
            OURS_Cycle_Error_Rate[key] = 0
        NCU_Cycle_Error_Rate[key] = 0

    MAPE_OURS = 0.

    num = 0
    for key in NCU_Cycle_Error_Rate.keys():
        if key in OURS_Cycle_Error_Rate.keys():
            MAPE_OURS += OURS_Cycle_Error_Rate[key]
            num += 1

    print('Cycle: MAPE_OURS:', MAPE_OURS/float(num))
    
    global MAPE_CYCLE_GLOBAL
    MAPE_CYCLE_GLOBAL = MAPE_OURS/float(num)

    from scipy.stats import pearsonr

    keys = NCU_Cycle.keys()
    assert OURS_Cycle.keys() == keys

    ncu_values = [NCU_Cycle[key] for key in keys]
    ours_values = [OURS_Cycle[key] for key in keys]

    ours_corr, _ = pearsonr(ncu_values, ours_values)

    print(f"Cycle: OURS - NCU Pearson's correlation coefficient: {ours_corr:.3f}")
    
    global CORR_CYCLE_GLOBAL
    CORR_CYCLE_GLOBAL = ours_corr

    return NCU_Cycle_Error_Rate, OURS_Cycle_Error_Rate

def plot_figure_Cycle_Error_Rate(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_GPU_Cycle_Error_Rate(
                Results_xlsx_path, 
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="OURS")
    
    NCU_Cycle_Error_Rate, OURS_Cycle_Error_Rate = prng[0], prng[1]

    NCU_Cycle_Error_Rate = {k: v for k, v in sorted(NCU_Cycle_Error_Rate.items(), key=lambda item: item[1], reverse=False)}
    OURS_Cycle_Error_Rate = {k: OURS_Cycle_Error_Rate[k] for k, v in sorted(NCU_Cycle_Error_Rate.items(), key=lambda item: item[1], reverse=False)}

    indexes = list(NCU_Cycle_Error_Rate.keys())
    
    
    prng = read_xlsx_L2_Hit_Rate(
                NCU_Cycle_Error_Rate.keys(),
                Results_xlsx_path, 
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="OURS")
    NCU_L2_Hit_Rate_Error_Rate, OURS_L2_Hit_Rate_Error_Rate = prng[0], prng[1]
    # print("OURS_L2_Hit_Rate_Error_Rate: ", OURS_L2_Hit_Rate_Error_Rate)
    # print(NCU_Cycle_Error_Rate.keys())
    
    prng = read_xlsx_L1_Hit_Rate(
                NCU_Cycle_Error_Rate.keys(),
                Results_xlsx_path, 
                NCU_sheet_name="NCU",
                PPT_sheet_name="PPT",
                ASIM_sheet_name="ASIM",
                OURS_sheet_name="OURS")
    NCU_L1_Hit_Rate_Error_Rate, OURS_L1_Hit_Rate_Error_Rate = prng[0], prng[1]
    # print("OURS_L1_Hit_Rate_Error_Rate: ", OURS_L1_Hit_Rate_Error_Rate)

    fig, axs = plt.subplots(ncols=1, nrows=1, num=style_label,
                            figsize=(5, 1.1), dpi=300) #layout='constrained', 

    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256
    
    OURS_Cycle_Error_Rate_list = [OURS_Cycle_Error_Rate[_] for _ in indexes]
    OURS_L1_Hit_Rate_Error_Rate = [OURS_L1_Hit_Rate_Error_Rate[_] for _ in indexes]
    OURS_L2_Hit_Rate_Error_Rate = [OURS_L2_Hit_Rate_Error_Rate[_] for _ in indexes]

    species = [_ for _ in indexes]
    penguin_means = {}
    
    global MAPE_CYCLE_GLOBAL
    global MAE_L1_GLOBAL
    global MAE_L2_GLOBAL

    global CORR_CYCLE_GLOBAL
    global CORR_L1_GLOBAL
    global CORR_L2_GLOBAL
    
    MAPE_CYCLE_GLOBAL = round(MAPE_CYCLE_GLOBAL, 4)
    MAE_L1_GLOBAL = round(MAE_L1_GLOBAL, 4)
    MAE_L2_GLOBAL = round(MAE_L2_GLOBAL, 4)
    
    CORR_CYCLE_GLOBAL = round(CORR_CYCLE_GLOBAL, 2)
    CORR_L1_GLOBAL = round(CORR_L1_GLOBAL, 2)
    CORR_L2_GLOBAL = round(CORR_L2_GLOBAL, 2)
    
    penguin_means['Cycles (MAPE: '+str(MAPE_CYCLE_GLOBAL*100)+'%, Corr.: '+str(CORR_CYCLE_GLOBAL)+')'] = tuple(float("%.2f" % float(OURS_Cycle_Error_Rate_list[_])) for _ in range(len(OURS_Cycle_Error_Rate_list)))
    penguin_means['L1 HR (MAE: '+str(MAE_L1_GLOBAL*100)+'%, Corr.: '+str(CORR_L1_GLOBAL)+')'] = tuple(float("%.2f" % float(OURS_L1_Hit_Rate_Error_Rate[_])) for _ in range(len(OURS_L1_Hit_Rate_Error_Rate)))
    penguin_means['L2 HR (MAE: '+str(MAE_L2_GLOBAL*100)+'%, Corr.: '+str(CORR_L2_GLOBAL)+')'] = tuple(float("%.2f" % float(OURS_L2_Hit_Rate_Error_Rate[_])) for _ in range(len(OURS_L2_Hit_Rate_Error_Rate)))


    x = np.arange(len(species))
    width = 0.25
    multiplier = 0
    color = ['#4eab90', '#c55a11', '#eebf6d']
    hatch = ['//', '\\\\', 'x']
    cluster_name = ['Cycles (MAPE: '+str(MAPE_CYCLE_GLOBAL*100)+'%, Corr.: '+str(CORR_CYCLE_GLOBAL)+')',\
                    'L1 HR (MAE: '+str(MAE_L1_GLOBAL*100)+'%, Corr.: '+str(CORR_L1_GLOBAL)+')',\
                    'L2 HR (MAE: '+str(MAE_L2_GLOBAL*100)+'%, Corr.: '+str(CORR_L2_GLOBAL)+')']
    
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
    
    axs.tick_params(axis='x', labelsize=8)
    axs.tick_params(axis='y', labelsize=8)
    axs.set_ylabel('Error Rate', fontsize=10)
    axs.set_ylim(0.0, .8)
    axs.set_title('')

    axs.legend(loc='lower center', fontsize=5.53, frameon=True, shadow=False, fancybox=False, framealpha=0.7, borderpad=0.3,
               ncol=3, markerfirst=True, markerscale=0.2, numpoints=1, handlelength=0.8, handleheight=0.4, bbox_to_anchor=(0.44, 1.02))
    
    indexes_short_name = [indexes_short_name_dict[_] for _ in indexes]

    axs.set_xticks(x)
    axs.set_xticklabels(indexes_short_name, rotation=90, fontsize=8)
    
    axs.grid(True, which='major', axis='y', linestyle='--', color='gray', linewidth=0.5)

if __name__ == "__main__":
    style_list = ['fast']

    for style_label in style_list:
        style_label_name = "classic"
        with plt.rc_context({"figure.max_open_warning": len(style_list)}):
            with plt.style.context(style_label):
                plot_figure_Cycle_Error_Rate(style_label=style_label)
                plt.savefig('figs/'+'fig_11_for_self_simulated_A100_results.pdf', format='pdf', bbox_inches='tight')
    

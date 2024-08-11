
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

Results_xlsx_path = "../Pre-Provided-Results/compare_for_pre_provided_GV100_results.xlsx"

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
    "LSTM_32",
    "GRU"
]

def read_xlsx_MemComStruct_Stalls(file_name="", OURS_sheet_name=""):
    
    MemFunctionalUnitPipelineSaturationCycles = {}
    MemFunctionalUnitIssuingMutualExclusionCycles = {}
    MemResultBusSaturationCycles = {}
    MemDispatchQueueSaturationCycles = {}
    MemBankConflictCycles = {}
    MemNoFreeOperandsCollectorUnitCycles = {}
    MemInterconnectInjectionBuferSaturationCycles = {}

    ComFunctionalUnitPipelineSaturationCycles = {}
    ComFunctionalUnitIssuingMutualExclusionCycles = {}
    ComResultBusSaturationCycles = {}
    ComDispatchQueueSaturationCycles = {}
    ComBankConflictCycles = {}
    ComNoFreeOperandsCollectorUnitCycles = {}
    
    None_of_OURS_System_Stalls = []

    data_OURS = pd.read_excel(file_name, sheet_name=OURS_sheet_name)

    for idx, row in data_OURS.iterrows():
        kernel_name = row["Unnamed: 0"]
        kernel_id = row["Kernel ID"]
        kernel_key = kernel_name
        
        kernel_MemFunctionalUnitPipelineSaturationCycles = row["MemoryStructuralStall_Issue_out_has_no_free_slot_Cycles"]
        kernel_MemFunctionalUnitIssuingMutualExclusionCycles = row["MemoryStructuralStall_Issue_previous_issued_inst_exec_type_is_memory_Cycles"]
        kernel_MemResultBusSaturationCycles = row["MemoryStructuralStall_Execute_result_bus_has_no_slot_for_latency_Cycles"]
        kernel_MemDispatchQueueSaturationCycles = row["MemoryStructuralStall_Execute_m_dispatch_reg_of_fu_is_not_empty_Cycles"]
        kernel_MemBankConflictCycles = row["MemoryStructuralStall_Writeback_bank_of_reg_is_not_idle_Cycles"]
        kernel_MemBankConflict1Cycles = row["MemoryStructuralStall_ReadOperands_bank_reg_belonged_to_was_allocated_Cycles"]
        kernel_MemNoFreeOperandsCollectorUnitCycles = row["MemoryStructuralStall_ReadOperands_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_Cycles"]
        kernel_MemInterconnectInjectionBuferSaturationCycles = row["MemoryStructuralStall_Execute_icnt_injection_buffer_is_full_Cycles"]

        kernel_ComFunctionalUnitPipelineSaturationCycles = row["ComputeStructuralStall_Issue_out_has_no_free_slot_Cycles"]
        kernel_ComFunctionalUnitIssuingMutualExclusionCycles = row["ComputeStructuralStall_Issue_previous_issued_inst_exec_type_is_compute_Cycles"]
        kernel_ComResultBusSaturationCycles = row["ComputeStructuralStall_Execute_result_bus_has_no_slot_for_latency_Cycles"]
        kernel_ComDispatchQueueSaturationCycles = row["ComputeStructuralStall_Execute_m_dispatch_reg_of_fu_is_not_empty_Cycles"]
        kernel_ComBankConflictCycles = row["ComputeStructuralStall_Writeback_bank_of_reg_is_not_idle_Cycles"]
        kernel_ComBankConflict1Cycles = row["ComputeStructuralStall_ReadOperands_bank_reg_belonged_to_was_allocated_Cycles"]
        kernel_ComNoFreeOperandsCollectorUnitCycles = row["ComputeStructuralStall_ReadOperands_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_Cycles"]


        
        if not kernel_key in Except_keys:
            if not kernel_key in MemFunctionalUnitPipelineSaturationCycles.keys():
                if isinstance(kernel_MemFunctionalUnitPipelineSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_MemFunctionalUnitPipelineSaturationCycles) and \
                    isinstance(kernel_MemFunctionalUnitIssuingMutualExclusionCycles, (int, float)) and \
                    not pd.isna(kernel_MemFunctionalUnitIssuingMutualExclusionCycles) and \
                    isinstance(kernel_MemResultBusSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_MemResultBusSaturationCycles) and \
                    isinstance(kernel_MemDispatchQueueSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_MemDispatchQueueSaturationCycles) and \
                    isinstance(kernel_MemBankConflictCycles, (int, float)) and \
                    not pd.isna(kernel_MemBankConflictCycles) and \
                    isinstance(kernel_MemNoFreeOperandsCollectorUnitCycles, (int, float)) and \
                    not pd.isna(kernel_MemNoFreeOperandsCollectorUnitCycles) and \
                    isinstance(kernel_MemInterconnectInjectionBuferSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_MemInterconnectInjectionBuferSaturationCycles) and \
                    isinstance(kernel_ComFunctionalUnitPipelineSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_ComFunctionalUnitPipelineSaturationCycles) and \
                    isinstance(kernel_ComFunctionalUnitIssuingMutualExclusionCycles, (int, float)) and \
                    not pd.isna(kernel_ComFunctionalUnitIssuingMutualExclusionCycles) and \
                    isinstance(kernel_ComResultBusSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_ComResultBusSaturationCycles) and \
                    isinstance(kernel_ComDispatchQueueSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_ComDispatchQueueSaturationCycles) and \
                    isinstance(kernel_ComBankConflictCycles, (int, float)) and \
                    not pd.isna(kernel_ComBankConflictCycles) and \
                    isinstance(kernel_ComNoFreeOperandsCollectorUnitCycles, (int, float)) and \
                    not pd.isna(kernel_ComNoFreeOperandsCollectorUnitCycles):
                    
                    MemFunctionalUnitPipelineSaturationCycles[kernel_key] = kernel_MemFunctionalUnitPipelineSaturationCycles
                    MemFunctionalUnitIssuingMutualExclusionCycles[kernel_key] = kernel_MemFunctionalUnitIssuingMutualExclusionCycles
                    MemResultBusSaturationCycles[kernel_key] = kernel_MemResultBusSaturationCycles
                    MemDispatchQueueSaturationCycles[kernel_key] = kernel_MemDispatchQueueSaturationCycles
                    MemBankConflictCycles[kernel_key] = kernel_MemBankConflictCycles + kernel_MemBankConflict1Cycles
                    MemNoFreeOperandsCollectorUnitCycles[kernel_key] = kernel_MemNoFreeOperandsCollectorUnitCycles
                    MemInterconnectInjectionBuferSaturationCycles[kernel_key] = kernel_MemInterconnectInjectionBuferSaturationCycles

                    ComFunctionalUnitPipelineSaturationCycles[kernel_key] = kernel_ComFunctionalUnitPipelineSaturationCycles
                    ComFunctionalUnitIssuingMutualExclusionCycles[kernel_key] = kernel_ComFunctionalUnitIssuingMutualExclusionCycles
                    ComResultBusSaturationCycles[kernel_key] = kernel_ComResultBusSaturationCycles
                    ComDispatchQueueSaturationCycles[kernel_key] = kernel_ComDispatchQueueSaturationCycles
                    ComBankConflictCycles[kernel_key] = kernel_ComBankConflictCycles + kernel_ComBankConflict1Cycles
                    ComNoFreeOperandsCollectorUnitCycles[kernel_key] = kernel_ComNoFreeOperandsCollectorUnitCycles
                else:
                    continue
            else:
                if isinstance(kernel_MemFunctionalUnitPipelineSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_MemFunctionalUnitPipelineSaturationCycles) and \
                    isinstance(kernel_MemFunctionalUnitIssuingMutualExclusionCycles, (int, float)) and \
                    not pd.isna(kernel_MemFunctionalUnitIssuingMutualExclusionCycles) and \
                    isinstance(kernel_MemResultBusSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_MemResultBusSaturationCycles) and \
                    isinstance(kernel_MemDispatchQueueSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_MemDispatchQueueSaturationCycles) and \
                    isinstance(kernel_MemBankConflictCycles, (int, float)) and \
                    not pd.isna(kernel_MemBankConflictCycles) and \
                    isinstance(kernel_MemNoFreeOperandsCollectorUnitCycles, (int, float)) and \
                    not pd.isna(kernel_MemNoFreeOperandsCollectorUnitCycles) and \
                    isinstance(kernel_MemInterconnectInjectionBuferSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_MemInterconnectInjectionBuferSaturationCycles) and \
                    isinstance(kernel_ComFunctionalUnitPipelineSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_ComFunctionalUnitPipelineSaturationCycles) and \
                    isinstance(kernel_ComFunctionalUnitIssuingMutualExclusionCycles, (int, float)) and \
                    not pd.isna(kernel_ComFunctionalUnitIssuingMutualExclusionCycles) and \
                    isinstance(kernel_ComResultBusSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_ComResultBusSaturationCycles) and \
                    isinstance(kernel_ComDispatchQueueSaturationCycles, (int, float)) and \
                    not pd.isna(kernel_ComDispatchQueueSaturationCycles) and \
                    isinstance(kernel_ComBankConflictCycles, (int, float)) and \
                    not pd.isna(kernel_ComBankConflictCycles) and \
                    isinstance(kernel_ComNoFreeOperandsCollectorUnitCycles, (int, float)) and \
                    not pd.isna(kernel_ComNoFreeOperandsCollectorUnitCycles):

                    MemFunctionalUnitPipelineSaturationCycles[kernel_key] += kernel_MemFunctionalUnitPipelineSaturationCycles
                    MemFunctionalUnitIssuingMutualExclusionCycles[kernel_key] += kernel_MemFunctionalUnitIssuingMutualExclusionCycles
                    MemResultBusSaturationCycles[kernel_key] += kernel_MemResultBusSaturationCycles
                    MemDispatchQueueSaturationCycles[kernel_key] += kernel_MemDispatchQueueSaturationCycles
                    MemBankConflictCycles[kernel_key] += kernel_MemBankConflictCycles + kernel_MemBankConflict1Cycles
                    MemNoFreeOperandsCollectorUnitCycles[kernel_key] += kernel_MemNoFreeOperandsCollectorUnitCycles
                    MemInterconnectInjectionBuferSaturationCycles[kernel_key] += kernel_MemInterconnectInjectionBuferSaturationCycles

                    ComFunctionalUnitPipelineSaturationCycles[kernel_key] += kernel_ComFunctionalUnitPipelineSaturationCycles
                    ComFunctionalUnitIssuingMutualExclusionCycles[kernel_key] += kernel_ComFunctionalUnitIssuingMutualExclusionCycles
                    ComResultBusSaturationCycles[kernel_key] += kernel_ComResultBusSaturationCycles
                    ComDispatchQueueSaturationCycles[kernel_key] += kernel_ComDispatchQueueSaturationCycles
                    ComBankConflictCycles[kernel_key] += kernel_ComBankConflictCycles + kernel_ComBankConflict1Cycles
                    ComNoFreeOperandsCollectorUnitCycles[kernel_key] += kernel_ComNoFreeOperandsCollectorUnitCycles
                else:
                    continue
        else:
            continue
    
    MemFunctionalUnitPipelineSaturationCycles_rate = {}
    MemFunctionalUnitIssuingMutualExclusionCycles_rate = {}
    MemResultBusSaturationCycles_rate = {}
    MemDispatchQueueSaturationCycles_rate = {}
    MemBankConflictCycles_rate = {}
    MemNoFreeOperandsCollectorUnitCycles_rate = {}
    MemInterconnectInjectionBuferSaturationCycles_rate = {}

    ComFunctionalUnitPipelineSaturationCycles_rate = {}
    ComFunctionalUnitIssuingMutualExclusionCycles_rate = {}
    ComResultBusSaturationCycles_rate = {}
    ComDispatchQueueSaturationCycles_rate = {}
    ComBankConflictCycles_rate = {}
    ComNoFreeOperandsCollectorUnitCycles_rate = {}

    for key in MemFunctionalUnitPipelineSaturationCycles.keys():
        # print("KEY: ", key)
        # print(MemFunctionalUnitPipelineSaturationCycles[key], MemFunctionalUnitIssuingMutualExclusionCycles[key], \
        #     MemResultBusSaturationCycles[key], MemDispatchQueueSaturationCycles[key], \
        #     MemBankConflictCycles[key], \
        #     MemNoFreeOperandsCollectorUnitCycles[key], MemInterconnectInjectionBuferSaturationCycles[key])
        all_Mem_cycles = float(MemFunctionalUnitPipelineSaturationCycles[key] + MemFunctionalUnitIssuingMutualExclusionCycles[key] + \
            MemResultBusSaturationCycles[key] + MemDispatchQueueSaturationCycles[key] + \
            MemBankConflictCycles[key] + \
            MemNoFreeOperandsCollectorUnitCycles[key] + MemInterconnectInjectionBuferSaturationCycles[key])
        if all_Mem_cycles != 0:
            MemFunctionalUnitPipelineSaturationCycles_rate[key] = \
                float(MemFunctionalUnitPipelineSaturationCycles[key]) / all_Mem_cycles
            MemFunctionalUnitIssuingMutualExclusionCycles_rate[key] = \
                float(MemFunctionalUnitIssuingMutualExclusionCycles[key]) / all_Mem_cycles
            MemResultBusSaturationCycles_rate[key] = \
                float(MemResultBusSaturationCycles[key]) / all_Mem_cycles
            MemDispatchQueueSaturationCycles_rate[key] = \
                float(MemDispatchQueueSaturationCycles[key]) / all_Mem_cycles
            MemBankConflictCycles_rate[key] = \
                float(MemBankConflictCycles[key]) / all_Mem_cycles
            MemNoFreeOperandsCollectorUnitCycles_rate[key] = \
                float(MemNoFreeOperandsCollectorUnitCycles[key]) / all_Mem_cycles
            MemInterconnectInjectionBuferSaturationCycles_rate[key] = \
                float(MemInterconnectInjectionBuferSaturationCycles[key]) / all_Mem_cycles
        else:
            MemFunctionalUnitPipelineSaturationCycles_rate[key] = 0.0
            MemFunctionalUnitIssuingMutualExclusionCycles_rate[key] = 0.0
            MemResultBusSaturationCycles_rate[key] = 0.0
            MemDispatchQueueSaturationCycles_rate[key] = 0.0
            MemBankConflictCycles_rate[key] = 0.0
            MemNoFreeOperandsCollectorUnitCycles_rate[key] = 0.0
            MemInterconnectInjectionBuferSaturationCycles_rate[key] = 0.0

        all_Com_cycles = float(ComFunctionalUnitPipelineSaturationCycles[key] + ComFunctionalUnitIssuingMutualExclusionCycles[key] + \
            ComResultBusSaturationCycles[key] + ComDispatchQueueSaturationCycles[key] + \
            ComBankConflictCycles[key] + \
            ComNoFreeOperandsCollectorUnitCycles[key])
        if all_Com_cycles != 0:
            ComFunctionalUnitPipelineSaturationCycles_rate[key] = \
                float(ComFunctionalUnitPipelineSaturationCycles[key]) / all_Com_cycles
            ComFunctionalUnitIssuingMutualExclusionCycles_rate[key] = \
                float(ComFunctionalUnitIssuingMutualExclusionCycles[key]) / all_Com_cycles
            ComResultBusSaturationCycles_rate[key] = \
                float(ComResultBusSaturationCycles[key]) / all_Com_cycles
            ComDispatchQueueSaturationCycles_rate[key] = \
                float(ComDispatchQueueSaturationCycles[key]) / all_Com_cycles
            ComBankConflictCycles_rate[key] = \
                float(ComBankConflictCycles[key]) / all_Com_cycles
            ComNoFreeOperandsCollectorUnitCycles_rate[key] = \
                float(ComNoFreeOperandsCollectorUnitCycles[key]) / all_Com_cycles
        else:
            ComFunctionalUnitPipelineSaturationCycles_rate[key] = 0.0
            ComFunctionalUnitIssuingMutualExclusionCycles_rate[key] = 0.0
            ComResultBusSaturationCycles_rate[key] = 0.0
            ComDispatchQueueSaturationCycles_rate[key] = 0.0
            ComBankConflictCycles_rate[key] = 0.0
            ComNoFreeOperandsCollectorUnitCycles_rate[key] = 0.0

    return MemFunctionalUnitPipelineSaturationCycles_rate, MemFunctionalUnitIssuingMutualExclusionCycles_rate, \
           MemResultBusSaturationCycles_rate, MemDispatchQueueSaturationCycles_rate, \
           MemBankConflictCycles_rate, \
           MemNoFreeOperandsCollectorUnitCycles_rate, MemInterconnectInjectionBuferSaturationCycles_rate, \
           ComFunctionalUnitPipelineSaturationCycles_rate, ComFunctionalUnitIssuingMutualExclusionCycles_rate, \
           ComResultBusSaturationCycles_rate, ComDispatchQueueSaturationCycles_rate, \
           ComBankConflictCycles_rate, \
           ComNoFreeOperandsCollectorUnitCycles_rate


def plot_figure_StackBar_System_Stalls(style_label=""):
    """Setup and plot the demonstration figure with a given style."""
    prng = read_xlsx_MemComStruct_Stalls(
                Results_xlsx_path, 
                OURS_sheet_name="OURS")
    
    OURS_DATA = None

    MemFunctionalUnitPipelineSaturationCycles_rate, MemFunctionalUnitIssuingMutualExclusionCycles_rate, \
    MemResultBusSaturationCycles_rate, MemDispatchQueueSaturationCycles_rate, \
    MemBankConflictCycles_rate, \
    MemNoFreeOperandsCollectorUnitCycles_rate, MemInterconnectInjectionBuferSaturationCycles_rate, \
    ComFunctionalUnitPipelineSaturationCycles_rate, ComFunctionalUnitIssuingMutualExclusionCycles_rate, \
    ComResultBusSaturationCycles_rate, ComDispatchQueueSaturationCycles_rate, \
    ComBankConflictCycles_rate, \
    ComNoFreeOperandsCollectorUnitCycles_rate = \
    prng[0], prng[1], prng[2], prng[3], prng[4], prng[5], prng[6], \
    prng[7], prng[8], prng[9], prng[10], prng[11], prng[12]

    indexes = list(MemFunctionalUnitPipelineSaturationCycles_rate.keys())
    
    background_color = mcolors.rgb_to_hsv(
        mcolors.to_rgb(plt.rcParams['figure.facecolor']))[2]
    if background_color < 0.5:
        title_color = [0.8, 0.8, 1]
    else:
        title_color = np.array([19, 6, 84]) / 256

    species = [_ for _ in indexes]
    penguin_means_Mem = {}
    penguin_means_Com = {}

    for key in MemFunctionalUnitPipelineSaturationCycles_rate.keys():
        penguin_means_Mem[key] =  ( float("%.6f" % float(MemFunctionalUnitPipelineSaturationCycles_rate[key])), \
                                    float("%.6f" % float(MemFunctionalUnitIssuingMutualExclusionCycles_rate[key])), \
                                    float("%.6f" % float(MemResultBusSaturationCycles_rate[key])), \
                                    float("%.6f" % float(MemDispatchQueueSaturationCycles_rate[key])), \
                                    float("%.6f" % float(MemBankConflictCycles_rate[key])), \
                                    float("%.6f" % float(MemNoFreeOperandsCollectorUnitCycles_rate[key])), \
                                    float("%.6f" % float(MemInterconnectInjectionBuferSaturationCycles_rate[key]))
                                  )
        penguin_means_Com[key] =  ( float("%.6f" % float(ComFunctionalUnitPipelineSaturationCycles_rate[key])), \
                                    float("%.6f" % float(ComFunctionalUnitIssuingMutualExclusionCycles_rate[key])), \
                                    float("%.6f" % float(ComResultBusSaturationCycles_rate[key])), \
                                    float("%.6f" % float(ComDispatchQueueSaturationCycles_rate[key])), \
                                    float("%.6f" % float(ComBankConflictCycles_rate[key])), \
                                    float("%.6f" % float(ComNoFreeOperandsCollectorUnitCycles_rate[key]))
                                  )
    
    x = np.arange(len(species))
    width = 0.3
    
    labels = list(penguin_means_Mem.keys())
    data_Mem = list(penguin_means_Mem.values())
    data_Com = list(penguin_means_Com.values())

    colors = ['#7fabd4', '#ff9f4e', '#7fc97f', '#d97d81', \
              '#e377c2', '#7f7f7f', '#bcbd22']
    colors = [
        '#fdae61',
        '#abd9e9',
        '#e6f598',
        '#fee08b',
        '#f4a582',
        '#91bfdb',
        '#f2b6b2'
    ]

    hatch = ['--', '//', '\\\\', 'xx', '++', '/|/|', '\\|\\|']

    fig, ax = plt.subplots(ncols=1, nrows=1, num=style_label,
                           figsize=(15, 2.3), layout='constrained', dpi=300)

    x = np.arange(len(labels))

    bottom_Mem = [0] * len(labels)
    bottom_Com = [0] * len(labels)
    
    for i in range(len(data_Mem[0])):
        offset = width + 0.1
        ax.bar(x, [d[i] for d in data_Mem], width=width, bottom=bottom_Mem, color=colors[i], hatch=hatch[i], edgecolor='#808080', linewidth=0.1)
        bottom_Mem = [sum(x) for x in zip(bottom_Mem, [d[i] for d in data_Mem])]
        if i < len(data_Com[0]):
            ax.bar(x+offset, [d[i] for d in data_Com], width=width, bottom=bottom_Com, color=colors[i], hatch=hatch[i], edgecolor='#808080', linewidth=0.1)
            bottom_Com = [sum(x) for x in zip(bottom_Com, [d[i] for d in data_Com])]
        

    ax.set_ylabel('Percentage')
    ax.set_title('Stacked Bar Chart')

    indexes_short_name = [indexes_short_name_dict[_] for _ in indexes]
    
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_ylabel('Mem/ComStruct Dist.', fontsize=14.5, y=0.3)
    ax.set_title('')
    

    import matplotlib.patches as mpatches
    legend_items = [mpatches.Patch(facecolor=colors[i], hatch=hatch[i], label=label, edgecolor='#808080', linewidth=0.1) 
                for i, label in enumerate([
                'Execution Unit Pipeline Saturation', \
                'Execution Unit Issuing Mutual Exclusion', \
                'Result Bus Saturation', \
                'Dispatch Queue Saturation', \
                'Bank Conflict', \
                'No Free Operands Collector Units', \
                'Interconnect Congestion'])]

    ax.legend(handles=legend_items, \
              fontsize=11, frameon=False, shadow=False, fancybox=False, \
              framealpha=1.0, borderpad=0.3, markerfirst=True, \
              markerscale=0.2, numpoints=1, handlelength=2.0, handleheight=0.5, \
              loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=4)
    
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
                plt.savefig('figs/'+'fig_13b_for_pre_provided_GV100_results.pdf', format='pdf')
    
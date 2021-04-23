import math
import matplotlib
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
import numpy as np
import re

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
dirbase = '../figures/'
logbase = '../logs/'
ourSys = 'HyQuas'
figsz = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 3),
}
plt.rcParams.update(figsz)
hb = '\\\\//\\\\//'

color_def = [
    '#f4b183',
    '#ffd966',
    '#c5e0b4',
    '#bdd7ee',
    "#8dd3c7",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#cccccc",
    "#fccde5",
    "#b3de69",
    "#ffd92f",
    "#66c2a4"
]

hatch_def = [
    '..',
    '//',
    '\\\\',
    'xx',
    '++',
    'oo',
    '..',
]

marker_def = [
    'o',
    '^',
    's',
    'd',
    '*',
    'D',
    'x',
    '+',
    '*',
    'v',
    '>',
    '2',
    '3'
]

fig_extension = 'pdf'
pg = 'PerGate'
bls = 'TransMM'
mix = 'Hybrid'
exectime = 'Exec. time (ms)'

def geoMean(lst):
    sum = 1
    for x in lst:
        sum *= x
    return math.pow(sum, 1/len(lst))

simName = {
    'basis_change': 'bc',
    'hidden_shift': 'hs',
    'quantum_volume': 'qv',
    'supremacy': 'sp'
}


def trans(labels):
    for i in range(len(labels)):
        if labels[i] in simName.keys():
            labels[i] = simName[labels[i]]
    return labels



def plot_single_gpu():
    # basis_change, bv, hidden_shift, qaoa, qft, quantum_volume, supremacy
    qcgpu_v100 = [22.648587465286255, 1.0285909175872803, 1.342470884323120, 9.01806902885437, 2.343612194061279, 8.310634136199951, 4.993691444396973]
    qcgpu_a100 = [13.230310916900635, 0.42561984062194824, 0.5957441329956055, 5.057258605957031, 1.1482579708099365, 4.636551856994629, 2.6947784423828125]
    qulacs_v100 = [4.653303750324994, 0.26372975390404463, 0.1442320728674531, 1.0468542012386024, 1.7434257962740958, 1.1854395898990333, 2.6002652649767697]
    qulacs_a100 = [2.047857452998869, 0.12407842103857547, 0.07454732398036867, 0.5061613629804924, 0.9289786290610209, 0.5753191190306097, 1.2552453679963946]
    qiskit_v100 = [7.849320411682129, 3.282461404800415, 3.10962176322937, 3.8705062866210938, 4.256688356399536, 3.9623775482177734, 4.732935905456543]
    qiskit_a100 = [3.8374032974243164, 0.9415936470031738, 0.8835525512695312, 1.3567233085632324, 1.5370030403137207, 1.4436469078063965, 1.8720848560333252]
    qibo_v100 = [46.505648136138916, 0.9364736080169678, 1.6403467655181885, 19.711951732635498, 1.68607759475708, 17.76024293899536, 8.932364463806152]
    qibo_a100 = [23.145111560821533, 0.47966957092285156, 0.8083326816558838, 9.912112712860107, 0.8943290710449219, 8.863016128540039, 4.479456186294556]
    yao_v100 = [54.977458407, 0.90499405, 1.77125664, 18.381935016, 3.20975295, 17.202068609, 10.158773486]
    yao_a100 = [28.736273299, 0.44781258, 0.913244741, 9.464926476, 1.519754208, 8.535676782, 5.552374209]
    quest_v100 = [37.767417, 0.802474, 1.355143, 16.548413, 1.854058, 15.117026, 7.533700]
    quest_a100 = [22.596018, 0.474108, 0.802901, 9.799473, 1.029840, 8.901999, 4.438175]
    my_v100 = []
    my_a100 = []
    # v100
    with open(logbase + "backend.log") as f:
        for st in f.readlines():
            if "mix" in st:
                my_v100.append(int(re.search('Time Cost: (\d+) us', st).group(1)) / 1e6)
    size = 7
    dat = [
        [x/y for x,y in zip(qcgpu_v100, my_v100)],
        [x/y for x,y in zip(qibo_v100, my_v100)],
        [x/y for x,y in zip(qiskit_v100, my_v100)],
        [x/y for x,y in zip(quest_v100, my_v100)],
        [x/y for x,y in zip(qulacs_v100, my_v100)],
        [x/y for x,y in zip(yao_v100, my_v100)],
        [x/y for x,y in zip(my_v100, my_v100)],
    ]
    for d in dat:
        d.append(geoMean(d))
    apps = ['bc', 'bv', 'hs', 'qaoa', 'qft', 'qv', 'sp', 'GeoMean']
    frameworks = ['QCGPU', 'Qibo', 'Qiskit', 'QuEST', 'Qulacs', 'Yao', 'HyQuas']
    num_apps = len(dat[0])
    num_bars = len(dat)
    print("[Report] v100 speedup: avg = {} max = {}".format(dat[4][-1], max(dat[4])))
    
    sz = (15, 5)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)

    color_vec = color_def[1:num_bars+1]
    hatch_vec = ['--', '..', '\\\\//', '\\\\', '//', 'o', '']

    fig, ax = plt.subplots()
    width, lgap, ggap, gggap = 0.5, 0, 0.75, 0
    ind = np.arange(num_apps) * (num_bars*(width+lgap) + gggap*(num_bars//2-1) + ggap)
    for i in range(num_bars):
        ax.bar(ind + (width+lgap)*i + gggap*(i>>1),
               dat[i], width, color=color_vec[i], hatch=hatch_vec[i], edgecolor='black')
    ax.set_ylim(0, 55)
    ax.set_yticks(np.arange(0, 6) * 10)
    plt.yticks(fontsize=20)
    plt.ylabel('Normalized exec. time', fontsize=25)
    plt.xticks(ind+(width+lgap+gggap/2)*3, apps, fontsize=25)
    legend_handles = [mpatches.Patch(
        facecolor=color_vec[i], hatch=hatch_vec[i], edgecolor='black') for i in range(num_bars)]
    plt.legend(legend_handles, frameworks, loc='upper center', ncol=4, fontsize=22, bbox_to_anchor=(0.5,1.32))
    fig.savefig(dirbase + 'v100-compare.pdf', bbox_inches='tight')
    # a100
    with open(logbase + "backend-a100.log") as f:
        for st in f.readlines():
            if "mix" in st:
                my_a100.append(int(re.search('Time Cost: (\d+) us', st).group(1)) / 1e6)
    size = 7
    dat = [
        [x/y for x,y in zip(qcgpu_a100, my_a100)],
        [x/y for x,y in zip(qibo_a100, my_a100)],
        [x/y for x,y in zip(qiskit_a100, my_a100)],
        [x/y for x,y in zip(quest_a100, my_a100)],
        [x/y for x,y in zip(qulacs_a100, my_a100)],
        [x/y for x,y in zip(yao_a100, my_a100)],
        [x/y for x,y in zip(my_a100, my_a100)],
    ]
    for d in dat:
        d.append(geoMean(d))
    print("[Report] a100 speedup: avg = {} max = {}".format(dat[4][-1], max(dat[4])))
    apps = ['bc', 'bv', 'hs', 'qaoa', 'qft', 'qv', 'sp', 'GeoMean']
    frameworks = ['QCGPU', 'Qibo', 'Qiskit', 'QuEST', 'Qulacs', 'Yao', 'HyQuas']
    num_apps = len(dat[0])
    num_bars = len(dat)

    sz = (15, 5)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)


    fig, ax = plt.subplots()
    width, lgap, ggap, gggap = 0.5, 0, 0.75, 0
    ind = np.arange(num_apps) * (num_bars*(width+lgap) + gggap*(num_bars//2-1) + ggap)
    for i in range(num_bars):
        ax.bar(ind + (width+lgap)*i + gggap*(i>>1),
               dat[i], width, color=color_vec[i], hatch=hatch_vec[i], edgecolor='black')
    ax.set_ylim(0, 30)
    ax.set_yticks(np.arange(0, 4) * 10)
    plt.yticks(fontsize=20)
    plt.ylabel('Normalized exec. time', fontsize=25)
    plt.xticks(ind+(width+lgap+gggap/2)*3, apps, fontsize=25)
    legend_handles = [mpatches.Patch(
        facecolor=color_vec[i], hatch=hatch_vec[i], edgecolor='black') for i in range(num_bars)]
    #plt.legend(legend_handles, frameworks, loc='upper center', ncol=4, fontsize=22, bbox_to_anchor=(0.5,1.32))
    fig.savefig(dirbase + 'a100-compare.pdf', bbox_inches='tight')

def plot_weak():
    times = []
    n_gates = set()
    with open(logbase + "weak_summary.log") as f:
        for st in f.readlines():
            t = re.search("Time Cost: (\d+)", st)
            if t is not None:
                times.append(int(t.group(1))/1e6)
            g = re.search("Total Gates (\d+)", st)
            if g is not None:
                n_gates.add(int(g.group(1)))
    
    gpu_1 = [times[0], times[3], times[6], times[9], times[12]]
    gpu_2 = [times[1], times[4], times[7], times[10], times[13], times[15]]
    gpu_4 = [times[2], times[5], times[8], times[11], times[14], times[16], times[17]]
    assert(len(times) == 18)

    dat = [gpu_1, gpu_2, gpu_4]
    labels = [
        '1 V100',
        '2 V100',
        '4 V100',
        'Weak scaling'
    ]
    num_type = len(dat)
    len_data = len(dat[0])
    
    x = list(range(24, 31))
    sz = (8, 3)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)

    color_vec = ['#f4a143',
                 '#ffc936',
                 '#78c679']
    marker_vec = marker_def

    fig, ax = plt.subplots()
    for i in range(num_type):
        ax.semilogy(list(range(24, 24 + len(dat[i]))), dat[i], color=color_vec[i], marker=marker_vec[i], markersize=10)
        
    plt.semilogy(list(range(24, 27)), [gpu_1[0], gpu_2[1], gpu_4[2]], "--", color = "#636363", label = "Weak Scaling")
    plt.semilogy(list(range(25, 28)), [gpu_1[1], gpu_2[2], gpu_4[3]], "--", color = "#636363")
    plt.semilogy(list(range(26, 29)), [gpu_1[2], gpu_2[3], gpu_4[4]], "--", color = "#636363")
    plt.semilogy(list(range(27, 30)), [gpu_1[3], gpu_2[4], gpu_4[5]], "--", color = "#636363")
    plt.semilogy(list(range(28, 31)), [gpu_1[4], gpu_2[5], gpu_4[6]], "--", color = "#636363")

    ax.set_xticks(x)
    legend_handles = [mlines.Line2D(
        [], [], color=color_vec[i], marker=marker_vec[i], label=labels[i], markersize=10) for i in range(num_type)]
    legend_handles.append(mlines.Line2D(
        [], [], color="#636363", ls='--', label="Weak Scaling"))
    plt.legend(loc='upper left', handles=legend_handles)
    plt.xlabel("Number of qubits", fontsize='large')
    plt.ylabel('Exec. time (s)', fontsize='large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_useOffset(False)

    fig.savefig(dirbase + 'v100-weak.pdf', bbox_inches='tight')


def plot_transmm_v100():
    time_28 = [[] for _ in range(11)]
    group_28 = [[] for _ in range(11)]
    blas_28 = [[] for _ in range(11)]
    cutt_28 = [[] for _ in range(11)]
    names = []
    with open(logbase + "transmm.log") as f:
        for s in f.readlines():
            t = re.search('-m(\d+)/(.*?)_(\d+).log:.*? Time Cost: (\d+) us', s)
            if t is not None:
                name = t.group(2)
                mat = int(t.group(1))
                qubit = int(t.group(3))
                us = int(t.group(4))
                time_28[mat].append(us)
            else:
                t = re.search('-m(\d+)/(.*?)_(\d+).log:.*? Total Groups: 1 (\d+)', s)
                mat = int(t.group(1))
                qubit = int(t.group(3))
                gg = int(t.group(4))
                group_28[mat].append(gg)
    
    time_28_geo = [geoMean(x) / 1000000 for x in time_28[3:]]
    group_28_geo = [geoMean(x) for x in group_28[3:]]
    with open(logbase + "transmm-profile.log") as f:
        cur_mat = -1
        cur_circ = -1
        cur_nqubit = -1
        blas = 0
        cutt = 0
        for st in f.readlines():
            if st[:5] == "+++++":
                if (cur_nqubit != -1):
                    blas_28[cur_mat].append(blas)
                    cutt_28[cur_mat].append(cutt)        
                cur_mat = int(st.strip().split()[1])
                cur_circ = -1
                cur_nqubit = -1
                blas = 0
                cutt = 0
            elif st[:5] == "=====":
                if (cur_nqubit != -1):
                    blas_28[cur_mat].append(blas)
                    cutt_28[cur_mat].append(cutt)        
                name = st.strip().split()[1]
                cur_circ = name
                cur_nqubit = int(name.split('_')[-1])
                blas = 0
                cutt = 0
            else:
                s = st.strip()[1:-1].replace('"', '').split(',')
                if s[7].startswith("volta_zgemm"):
                    blas += float(s[1]) / 100
                elif s[7].startswith("void transpose"):
                    cutt += float(s[1]) / 100
                else:
                    assert(False)
        blas_28[cur_mat].append(blas)
        cutt_28[cur_mat].append(cutt)        
        
    # print(cutt_25, [len(x) for x in cutt_25])
    # print(cutt_28, [len(x) for x in cutt_28])
    # print(blas_25, [len(x) for x in blas_25])
    # print(blas_28, [len(x) for x in blas_28])
    
    blas_28_min = [min([xx for xx, yy in zip(x,y)]) for x, y in zip(blas_28[3:], time_28[3:])]
    blas_28_max = [max([xx for xx, yy in zip(x,y)]) for x, y in zip(blas_28[3:], time_28[3:])]
    blas_28_geo = [geoMean([xx for xx, yy in zip(x,y)]) for x, y in zip(blas_28[3:], time_28[3:])]
    cutt_28_min = [min([xx for xx, yy in zip(x,y)]) for x, y in zip(cutt_28[3:], time_28[3:])]
    cutt_28_max = [max([xx for xx, yy in zip(x,y)]) for x, y in zip(cutt_28[3:], time_28[3:])]
    cutt_28_geo = [geoMean([xx for xx, yy in zip(x,y)]) for x, y in zip(cutt_28[3:], time_28[3:])]
    blas_28_geo = [x * y for x, y in zip(blas_28_geo, time_28_geo)]
    blas_28_min = [x * y for x, y in zip(blas_28_min, time_28_geo)]
    blas_28_max = [x * y for x, y in zip(blas_28_max, time_28_geo)]
    cutt_28_geo = [x * y for x, y in zip(cutt_28_geo, time_28_geo)]
    cutt_28_min = [x * y for x, y in zip(cutt_28_min, time_28_geo)]
    cutt_28_max = [x * y for x, y in zip(cutt_28_max, time_28_geo)]
    circ_id = -1

    blas_28_min = [x - y for x, y in zip(blas_28_geo, blas_28_min)]
    blas_28_max = [y - x for x, y in zip(blas_28_geo, blas_28_max)]
    cutt_28_min = [x - y for x, y in zip(cutt_28_geo, cutt_28_min)]
    cutt_28_max = [y - x for x, y in zip(cutt_28_geo, cutt_28_max)]

    return blas_28_geo, cutt_28_geo, group_28_geo


def plot_transmm_a100():
    time_28 = [[] for _ in range(11)]
    group_28 = [[] for _ in range(11)]
    blas_28 = [[] for _ in range(11)]
    cutt_28 = [[] for _ in range(11)]
    names = []
    with open(logbase + "transmm-a100.log") as f:
        for s in f.readlines():
            t = re.search('-m(\d+)/(.*?)_(\d+).log:.*? Time Cost: (\d+) us', s)
            if t is not None:
                name = t.group(2)
                mat = int(t.group(1))
                qubit = int(t.group(3))
                us = int(t.group(4))
                if mat == 3:
                    names.append(name)
                time_28[mat].append(us)
            else:
                t = re.search('-m(\d+)/(.*?)_(\d+).log:.*? Total Groups: 1 (\d+)', s)
                mat = int(t.group(1))
                qubit = int(t.group(3))
                gg = int(t.group(4))
                group_28[mat].append(gg)
    
    time_28_geo = [geoMean(x) / 1000000 for x in time_28[3:]]
    group_28_geo = [geoMean(x) for x in group_28[3:]]
    with open(logbase + "transmm-profile-a100.log") as f:
        cur_mat = -1
        cur_circ = -1
        cur_nqubit = -1
        blas = 0
        cutt = 0
        for st in f.readlines():
            if st[:5] == "+++++":
                if (cur_nqubit != -1):
                    if cur_nqubit == 25:
                        blas_25[cur_mat].append(blas)
                        cutt_25[cur_mat].append(cutt)
                    elif cur_nqubit == 28:
                        blas_28[cur_mat].append(blas)
                        cutt_28[cur_mat].append(cutt)        
                    else:
                        assert(cur_nqubit == 30)
                cur_mat = int(st.strip().split()[1])
                cur_circ = -1
                cur_nqubit = -1
                blas = 0
                cutt = 0
            elif st[:5] == "=====":
                if (cur_nqubit != -1):
                    if cur_nqubit == 25:
                        blas_25[cur_mat].append(blas)
                        cutt_25[cur_mat].append(cutt)
                    elif cur_nqubit == 28:
                        blas_28[cur_mat].append(blas)
                        cutt_28[cur_mat].append(cutt)        
                    else:
                        assert(cur_nqubit == 30)
                name = st.strip().split()[1]
                cur_circ = name
                cur_nqubit = int(name.split('_')[-1])
                blas = 0
                cutt = 0
            else:
                s = st.strip().split()
                if s[7].startswith("cutlass"):
                    blas += int(s[1])
                elif s[7].startswith("transpose"):
                    cutt += int(s[1])
                else:
                    assert(False)
        if cur_nqubit == 25:
            blas_25[cur_mat].append(blas)
            cutt_25[cur_mat].append(cutt)
        elif cur_nqubit == 28:
            blas_28[cur_mat].append(blas)
            cutt_28[cur_mat].append(cutt)        
        else:
            assert(cur_nqubit == 30)
            
    blas_28_min = [min([xx/yy for xx, yy in zip(x,y)]) for x, y in zip(blas_28[3:], time_28[3:])]
    blas_28_max = [max([xx/yy for xx, yy in zip(x,y)]) for x, y in zip(blas_28[3:], time_28[3:])]
    blas_28_geo = [geoMean([xx/yy for xx, yy in zip(x,y)]) for x, y in zip(blas_28[3:], time_28[3:])]
    cutt_28_min = [min([xx/yy for xx, yy in zip(x,y)]) for x, y in zip(cutt_28[3:], time_28[3:])]
    cutt_28_max = [max([xx/yy for xx, yy in zip(x,y)]) for x, y in zip(cutt_28[3:], time_28[3:])]
    cutt_28_geo = [geoMean([xx/yy for xx, yy in zip(x,y)]) for x, y in zip(cutt_28[3:], time_28[3:])]
    blas_28_geo = [x * y / 1000 for x, y in zip(blas_28_geo, time_28_geo)]
    blas_28_min = [x * y / 1000 for x, y in zip(blas_28_min, time_28_geo)]
    blas_28_max = [x * y / 1000 for x, y in zip(blas_28_max, time_28_geo)]
    cutt_28_geo = [x * y / 1000 for x, y in zip(cutt_28_geo, time_28_geo)]
    cutt_28_min = [x * y / 1000 for x, y in zip(cutt_28_min, time_28_geo)]
    cutt_28_max = [x * y / 1000 for x, y in zip(cutt_28_max, time_28_geo)]
    circ_id = -1
    blas_28_min = [x - y for x, y in zip(blas_28_geo, blas_28_min)]
    blas_28_max = [y - x for x, y in zip(blas_28_geo, blas_28_max)]
    cutt_28_min = [x - y for x, y in zip(cutt_28_geo, cutt_28_min)]
    cutt_28_max = [y - x for x, y in zip(cutt_28_geo, cutt_28_max)]

    return blas_28_geo, cutt_28_geo, group_28_geo


def plot_transmm():
    v100blas_28_geo, v100cutt_28_geo, v100group_28_geo = plot_transmm_v100()
    a100blas_28_geo, a100cutt_28_geo, a100group_28_geo = plot_transmm_a100()

    sz = (15, 5)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)
    color_vec = color_def[1:]
    num_bars = 2
    apps = list(range(3, 11))

    fig, axes = plt.subplots(ncols=2)
    ax1, ax2 = axes[0], axes[1]

    ax = ax1
    mat_size = list(range(3, 11))
    x = np.array(list(range(len(mat_size))))
    ax.bar(x * 2, v100blas_28_geo, bottom = v100cutt_28_geo,  width = 1, label = "GEMM", color = color_vec[0], hatch = hatch_def[0], alpha=.99)
    ax.bar(x * 2, v100cutt_28_geo, width = 1, label = "Transpose", color = color_vec[1], hatch = hatch_def[1], alpha=.99)
    #plt.xticks(x * 2, mat_size, fontsize='large')
    #plt.xlabel('Active Qubit Size', fontsize='large')
    ax.set_ylabel('Exec. time (s)', fontsize=20)
    ax.legend(ncol=2, loc = "upper center", bbox_to_anchor=(0.5,1.2),fontsize=20)
    bx = ax.twinx()
    bx.plot(x * 2, v100group_28_geo, label = 'Merged Gates', color = 'r')
    #bx.set_ylabel('Merged Gate Count')
    bx.legend(loc = "upper center", bbox_to_anchor=(1.8,1.2),fontsize=20)
    plt.setp(bx.get_yticklabels(), fontsize=20, color = 'r')
    #ax.set_ylim(0, 1.4)
    #bx.set_ylim(10, 80)

    color_vec = color_def[1:num_bars+1]

    #fig, ax = plt.subplots()
    ax = ax2
    mat_size = list(range(3, 11))
    x = np.array(list(range(len(mat_size))))
    ax.bar(x * 2, a100blas_28_geo, bottom = a100cutt_28_geo,  width = 1, label = "GEMM", color = color_vec[0], hatch = hatch_def[0], alpha=.99)
    ax.bar(x * 2, a100cutt_28_geo, width = 1, label = "Transpose", color = color_vec[1], hatch = hatch_def[1], alpha=.99)
    #plt.xticks(x * 2, mat_size, fontsize='large')
    #plt.xlabel('Active Qubit Size', fontsize='large')
    #ax.set_ylabel(exectime)
    #ax.legend(ncol=2, loc = "upper center", bbox_to_anchor=(0.3,1.25))
    #ax.set_yticks([])
    bx = ax.twinx()
    bx.plot(x * 2, a100group_28_geo, label = 'Merged Gates', color = 'r')
    bx.set_ylabel('Merged gate count', fontsize=20, color = 'r')
    plt.setp(bx.get_yticklabels(), fontsize=20, color = 'r')
    #bx.legend(loc = "upper center", bbox_to_anchor=(0.8,1.25))
    #ax.set_ylim(0, 1.4)
    #bx.set_ylim(10, 80)
    #plt.xlabel("Merged gate size", fontsize='large')
    #plt.ylabel('Exec. Time (s)', fontsize='large')
    #plt.xticks(ind+(width+lgap+gggap/2)*0.5, trans(apps))
    legend_handles = [mpatches.Patch(
        facecolor=color_vec[i], edgecolor='black', hatch=hatch_def[i]) for i in range(num_bars)]
    #plt.legend(legend_handles, frameworks, loc='upper left', ncol=3)
    #plt.xticks(fontsize='large')


    plt.setp(ax1.get_xticklabels(), fontsize=20)
    plt.setp(ax1.get_yticklabels(), fontsize=20)
    plt.setp(ax2.get_xticklabels(), fontsize=20)
    plt.setp(ax2.get_yticklabels(), fontsize=20)
    #plt.setp(ax1.twinx().get_yticklabels(), fontsize=25)
    #plt.setp(ax2.twinx().get_yticklabels(), fontsize=25)
    ax1.set_xticks(x*2)
    ax1.set_xticklabels(trans(apps))
    ax2.set_xticks(x*2)
    ax2.set_xticklabels(trans(apps))
    # plt.xticks(ind+(width+lgap+gggap/2), trans(apps))

    ax1.set_title('(a) V100',y=-0.2, fontsize=25)
    ax2.set_title('(b) A100',y=-0.2, fontsize=25)


    #legend_handles = [mpatches.Patch(
    #    facecolor=color_vec[i], edgecolor='black', hatch=hatch_def[i]) for i in range(num_bars)]
    #plt.legend(legend_handles, frameworks, loc='upper left', ncol=3, fontsize=20, bbox_to_anchor=(-1.12,1.2))

    fig.subplots_adjust(wspace=0.2)
    plt.savefig(dirbase + 'two-transmm.pdf', bbox_inches='tight')

def plot_scale_v100():
    time_28 = {
        '1gpu-o': [],
        '1gpu-s': [],
        '2gpu-o': [],
        '2gpu-s': [],
        '4gpu-o': [],
        '4gpu-s': []
    }
    names = []
    with open(logbase + "scale.log") as f:
        for s in f.readlines():
            t = re.search('-(\dgpu-[so])/(.*?)_(\d+).log:.*? Time Cost: (\d+) us', s)
            name = t.group(2)
            exp = t.group(1)
            qubit = int(t.group(3))
            us = int(t.group(4))
            if exp == '1gpu-o':
                names.append(name)
            time_28[exp].append(us / 1000000)
            
    dat = []
    dat.append([1 for _ in time_28['1gpu-s']])
    dat.append([y / x for x, y in zip(time_28['2gpu-s'], time_28['1gpu-s'])])
    dat.append([y / x for x, y in zip(time_28['2gpu-o'], time_28['1gpu-s'])])
    dat.append([y / x for x, y in zip(time_28['4gpu-s'], time_28['1gpu-s'])])
    dat.append([y / x for x, y in zip(time_28['4gpu-o'], time_28['1gpu-s'])])
    labels = ['1 V100', '2 V100 ', '2 V100 (overlap)', '4 V100 ', '4 V100 (overlap)']
    
    apps = names
    frameworks = labels
    num_apps = len(dat[0])
    num_bars = len(dat)

    sz = (9, 4)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)

    color_vec = color_def[1:num_bars+1]

    fig, ax = plt.subplots()
    width, lgap, ggap, gggap = 2, 0, 3, 0.8
    ind = np.arange(num_apps) * (num_bars*(width+lgap) + gggap*(num_bars//2-1) + ggap)
    for i in range(num_bars):
        ax.bar(ind + (width+lgap)*i + gggap*(i+1>>1),
               dat[i], width, color=color_vec[i], hatch=hatch_def[i], edgecolor='black')
    ax.set_ylim(0, 3.8)
    ax.set_yticks(np.arange(0, 4))
    plt.yticks(fontsize='xx-large')
    plt.ylabel('Speedup', fontsize='x-large')
    plt.xticks(ind+(width+lgap+gggap/2)*2, trans(apps), fontsize='x-large')
    legend_handles = [mpatches.Patch(
        facecolor=color_vec[i], edgecolor='black', hatch=hatch_def[i]) for i in range(num_bars)]
    n_frame = frameworks
    legend_handles = [legend_handles[0], mpatches.Patch(fill='false', facecolor='white')] + legend_handles[1:]
    n_frame = [n_frame[0], ''] + n_frame[1:]
    leg = plt.legend(legend_handles, n_frame, loc='upper center', ncol=3, bbox_to_anchor=(0.5,1.32), fontsize='x-large')
    fig.savefig(dirbase + 'v100-scale.pdf', bbox_inches='tight')
    
def plot_pergate_v100():
    baseline_c = []
    baseline_n = []
    multitask_c = []
    multitask_n = []
    lookup_c = []
    lookup_n = []
    bank_c = []
    bank_n = []
    name = []
    with open(logbase + "pergate.log") as f:
        f.readline()
        for iii in range(14):
            st = f.readline().strip().split()
            t0 = [int(x) for x in st[1:4]]
            t1 = [int(x) for x in st[4:10]]
            name.append(st[0][:-1])
            baseline_c.append(sum(t0)/len(t0)/512)
            baseline_n.append(sum(t1)/len(t1)/512)
        f.readline()
        for iii in range(14):
            st = f.readline().strip().split()
            t0 = [int(x) for x in st[1:4]]
            t1 = [int(x) for x in st[4:10]]
            multitask_c.append(sum(t0)/len(t0)/512)
            multitask_n.append(sum(t1)/len(t1)/512)
        f.readline()
        for iii in range(14):
            st = f.readline().strip().split()
            t0 = [int(x) for x in st[1:4]]
            t1 = [int(x) for x in st[4:10]]
            lookup_c.append(sum(t0)/len(t0)/512)
            lookup_n.append(sum(t1)/len(t1)/512)
        f.readline()
        for iii in range(14):
            st = f.readline().strip().split()
            t0 = [int(x) for x in st[1:4]]
            t1 = [int(x) for x in st[4:10]]
            bank_c.append(sum(t0)/len(t0)/512)
            bank_n.append(sum(t1)/len(t1)/512)
    old_dat = [baseline_c, baseline_n, multitask_c, multitask_n, lookup_c, lookup_n, bank_c, bank_n]
    dat = []
    err  = 0
    for dt in old_dat:
        dat.append([(dt[0] + dt[6])/2, (dt[1] + dt[2]) / 2, (dt[3] + dt[5]) / 2, dt[4], (dt[7] + dt[8]) / 2, (dt[9] + dt[10]) / 2, (dt[11] + dt[12] + dt[13]) / 3])
        err = max([err, dt[0] - dt[6]], key = abs)
        err = max([err, dt[1] - dt[2]], key = abs)
        err = max([err, dt[3] - dt[5]], key = abs)
        err = max([err, dt[7] - dt[8]], key = abs)
        err = max([err, dt[9] - dt[10]], key = abs)
        err = max([err, dt[11] - dt[12], dt[11] - dt[13], dt[12] - dt[13]], key = abs)

    
    apps = ['U1/Z', 'U2/U3', 'H/Y', name[4], 'S/SDG', 'T/TDG', 'RX/RY/RZ']
    frameworks = []
    for x in ['baseline', 'multitask', 'lookup', 'bank']:
        for y in ['0-2', '3-9']:
            frameworks.append('%s (qubit %s)' % (x, y))
    num_apps = len(dat[0])
    num_bars = len(dat)

    sz = (25, 7)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)

    color_vec = [color_def[1], color_def[-1], color_def[1], color_def[-1], color_def[1], color_def[-1], color_def[1], color_def[-1]]
    hatch_vec = ['.', '.', '/', '/', '\\', '\\', '', '']

    fig, ax = plt.subplots()
    width, lgap, ggap, gggap = 0.5, 0, 0.75, 0.2
    ind = np.arange(num_apps) * (num_bars*(width+lgap) + gggap*(num_bars//2-1) + ggap)
    for i in range(num_bars):
        ax.bar(ind + (width+lgap)*i + gggap*(i>>1),
               dat[i], width, color=color_vec[i], hatch=hatch_vec[i], edgecolor='black')
    x_pos = ind + (width+lgap)*i + gggap*(i>>1)
    plt.xlim(min(x_pos) - 5, max(x_pos) + 1)
    ax.set_ylim(0, 1700)
    ax.set_yticks(list(np.arange(0, 4) * 500) + [1700])
    plt.yticks(fontsize=30)
    plt.ylabel('Time per gate (us)', fontsize=35)
    plt.xticks(ind+(width+lgap+gggap/2)*3.5, apps, fontsize=33)
    legend_handles = [mpatches.Patch(
        facecolor=color_vec[i], hatch=hatch_vec[i], edgecolor='black') for i in range(num_bars)]
    plt.legend(legend_handles, frameworks, loc='upper center', ncol=4, fontsize=28, bbox_to_anchor=(0.45,1.35))
    fig.savefig(dirbase + 'v100-pergate.pdf', bbox_inches='tight')

    
def plot_groupsz():
    sz = (15, 5)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)

    fig, axes = plt.subplots(ncols=2)
    ax1, ax2 = axes[0], axes[1]

    qulacs_a100 = []
    qulacs_v100 = []
    my_a100 = []
    my_v100 = []

    with open(logbase + "groupsz-bv-a100.log") as f:
        for i in range(3, 11):
            st = f.readline().strip().split()
            qulacs_a100.append(float(st[-1]) / 2)
    with open(logbase + "groupsz-bv.log") as f:
        for i in range(3, 11):
            st = f.readline().strip().split()
            qulacs_v100.append(float(st[-1]) / 2)
    with open(logbase + "groupsz-tm-a100.log") as f:
        for i in range(3, 11):
            st = f.readline().strip().split()
            tm = [int(x)/(1e6) for x in st[2:]]
            my_a100.append(sum(tm) / len(tm) / 2)
    with open(logbase + "groupsz-tm.log") as f:
        for i in range(3, 11):
            st = f.readline().strip().split()
            tm = [int(x)/(1e6) for x in st[2:]]
            my_v100.append(sum(tm) / len(tm) / 2)

    dat = [qulacs_v100, my_v100]
    labels = ['BatchMV', 'TransMM']

    apps = list(range(3, 11))
    frameworks = labels
    num_apps = len(dat[0])
    num_bars = len(dat)
    color_vec = color_def[1:num_bars+1]
    hatch_vec = ['..', '']

    ax = ax1
    width, lgap, ggap, gggap = 1.5, 0, 3, 0
    ind = np.arange(num_apps) * (num_bars*(width+lgap) + gggap*(num_bars//2-1) + ggap)
    for i in range(num_bars):
        ax.bar(ind + (width+lgap)*i + gggap*(i>>1),
            dat[i], width, color=color_vec[i], hatch=hatch_vec[i], edgecolor='black')
    ax.set_ylim(0, 3)
    ax.set_yticks(np.arange(0, 4))
    #plt.xlabel("Merged gate size", fontsize='large')
    #plt.ylabel('Exec. time (s)', fontsize='large')
    #plt.xticks(ind+(width+lgap+gggap/2)*0.5, trans(apps))
    # plt.xticks(fontsize='large')
    # plt.show()
    # fig.savefig('blas-group.pdf', bbox_inches='tight')

    dat = [qulacs_a100, my_a100]
    labels = ['BatchMV', 'TransMM']
    print("[Report] groupsz speedup 6 {} 10 {}".format(qulacs_a100[3]/my_a100[3], qulacs_a100[-1]/my_a100[-1]))

    apps = list(range(3, 11))
    frameworks = labels
    num_apps = len(dat[0])
    num_bars = len(dat)

    sz = (5, 3)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)

    color_vec = color_def[1:num_bars+1]

    #fig, ax = plt.subplots()
    ax = ax2
    width, lgap, ggap, gggap = 1.5, 0, 3, 0
    ind = np.arange(num_apps) * (num_bars*(width+lgap) + gggap*(num_bars//2-1) + ggap)
    for i in range(num_bars):
        ax.bar(ind + (width+lgap)*i + gggap*(i>>1),
            dat[i], width, color=color_vec[i], hatch=hatch_vec[i], edgecolor='black')
    ax.set_ylim(0, 3)
    ax.set_yticks(np.arange(0, 4))
    #plt.xlabel("Merged gate size", fontsize='large')
    ax1.set_ylabel('Exec. time (s)', fontsize=25)
    #plt.xticks(ind+(width+lgap+gggap/2)*0.5, trans(apps))
    legend_handles = [mpatches.Patch(
        facecolor=color_vec[i], edgecolor='black', hatch=hatch_vec[i]) for i in range(num_bars)]
    #plt.legend(legend_handles, frameworks, loc='upper left', ncol=3)
    #plt.xticks(fontsize='large')


    plt.setp(ax1.get_xticklabels(), fontsize=25)
    plt.setp(ax1.get_yticklabels(), fontsize=25)
    plt.setp(ax2.get_xticklabels(), fontsize=25)
    plt.setp(ax2.get_yticklabels(), fontsize=25)
    ax1.set_xticks(ind+(width+lgap+gggap/2)-0.75)
    ax1.set_xticklabels(trans(apps))
    ax2.set_xticks(ind+(width+lgap+gggap/2)-0.75)
    ax2.set_xticklabels(trans(apps))
    # plt.xticks(ind+(width+lgap+gggap/2), trans(apps))

    ax1.set_xlabel("Merged gate size", fontsize=25)
    ax2.set_xlabel("Merged gate size", fontsize=25)

    ax1.set_title('(a) V100',y=-0.35, fontsize=25)
    ax2.set_title('(b) A100',y=-0.35, fontsize=25)


    legend_handles = [mpatches.Patch(
        facecolor=color_vec[i], edgecolor='black', hatch=hatch_vec[i]) for i in range(num_bars)]
    plt.legend(legend_handles, frameworks, loc='upper left', ncol=3, fontsize=20, bbox_to_anchor=(-0.52,1.25))

    fig.subplots_adjust(wspace=0.12)
    plt.savefig(dirbase + 'two-groupsz.pdf', bbox_inches='tight')

def plot_numgate():
    blas_perf = []
    group_perf = []
    transmm_perf = []
    x = []
    with open(logbase + "numgate-sm-a100.log") as f:
        for i in range(33):
            n, tm = f.readline().strip().split(":")
            n = int(n)
            tm = [int(x) / 1000 for x in tm.split()]
            tm = sum(tm) * 1.0 / len(tm)
            x.append(n)
            group_perf.append(tm)
    with open(logbase + "numgate-bv-a100.log") as f:
        for i in range(33):
            n, tm = f.readline().strip().split()
            n = int(n)
            tm = float(tm) * 1000
            blas_perf.append(tm)
    with open(logbase + "numgate-tm-a100.log") as f:
        for i in range(33):
            n, tm = f.readline().strip().split(":")
            n = int(n)
            tm = [int(x) / 1000 for x in tm.split()]
            tm = sum(tm) * 1.0 / len(tm)
            transmm_perf.append(tm)
    print("[Report]: TransMM/BatchMV", sum(blas_perf) / sum(transmm_perf))
    dat = [group_perf, blas_perf]
    labels = ['ShareMem', 'BatchMV']
    num_type = len(dat)
    len_data = len(dat[0])

    sz = (6, 2)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)

    color_vec = ['#f4a143',
                 '#78c679']
    marker_vec = ['o', 'x']

    fig, ax = plt.subplots()
    for i in range(num_type):
        ax.plot(x, dat[i], color=color_vec[i], marker=marker_vec[i])
    plt.vlines(132, 0, dat[0][20], colors = "#636363", linestyles='dashed')
    ax.set_xticks([25*x for x in range(9)])
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    legend_handles = [mlines.Line2D(
        [], [], color=color_vec[i], label=labels[i], marker=marker_vec[i]) for i in range(num_type)]
    plt.legend(loc='lower right', handles=legend_handles)
    plt.xlabel('Number of gate')
    plt.ylabel(exectime)

    plt.show()
    fig.savefig(dirbase + 'a100-numgate.pdf', bbox_inches='tight')

def plot_backend():
    sz = (15, 5)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)

    fig, axes = plt.subplots(ncols=2)
    ax1, ax2 = axes[0], axes[1]

    # v100
    time_28 = {
        'blas': [],
        'group': [],
        'mix': []
    }
    names = []
    with open(logbase + "backend.log") as f:
        for s in f.readlines():
            t = re.search('-([a-z]+)/(.*?)_(\d+).log:.*? Time Cost: (\d+) us', s)
            if t is not None:
                name = t.group(2)
                backend = t.group(1)
                qubit = int(t.group(3))
                us = int(t.group(4))
                if backend == 'blas':
                    names.append(name)
                time_28[backend].append(us / 1000000)

    time_28['group'].append(geoMean(time_28['group']))
    time_28['blas'].append(geoMean(time_28['blas']))
    time_28['mix'].append(geoMean(time_28['mix']))
    names.append('GeoMean')
    
    dat = [time_28['group'], time_28['blas'], time_28['mix']]
    apps = names
    frameworks = ['OShareMem', 'TransMM', 'Hybrid']
    num_apps = len(dat[0])
    num_bars = len(dat)
    color_vec = color_def[1:num_bars+1]
    hatch_vec = ['..', '//', '']
    
    # ax1.set_xticklabels(fontsize=18)
    # ax1.set_yticklabels(fontsize=18)
    # ax2.set_xticklabels(fontsize=18)
    # ax2.set_yticklabels(fontsize=18)

    ax1.set_ylabel("Exec. time (s)", fontsize=25)
    
    ax = ax1
    width, lgap, ggap, gggap = 1.5, 0, 2, 0
    ind = np.arange(num_apps) * (num_bars*(width+lgap) + gggap*(num_bars//2-1) + ggap)
    for i in range(num_bars):
        ax.bar(ind + (width+lgap)*i + gggap*(i>>1),
            dat[i], width, color=color_vec[i], hatch=hatch_vec[i], edgecolor='black')
    ax.set_ylim(0, 3.5)
    ax.set_yticks(np.arange(0, 4))

    print("[Report] mix speedup (osm, tm) v100 avg {} {} max {} {} qv {} {}".format(
        time_28['group'][-1]/time_28['mix'][-1],
        time_28['blas'][-1] / time_28['mix'][-1],
        max([x/y for x,y in zip(time_28['group'], time_28['mix'])]),
        max([x/y for x,y in zip(time_28['blas'], time_28['mix'])]),
        time_28['group'][5]/time_28['mix'][5],
        time_28['blas'][5] / time_28['mix'][5]
    ))

    # A100
    time_28 = {
        'blas': [],
        'group': [],
        'mix': []
    }
    names = []
    with open(logbase + "backend-a100.log") as f:
        for s in f.readlines():
            t = re.search('-([a-z]+)/(.*?)_(\d+).log:.*? Time Cost: (\d+) us', s)
            if t is not None:
                name = t.group(2)
                backend = t.group(1)
                qubit = int(t.group(3))
                us = int(t.group(4))
                if backend == 'blas':
                    names.append(name)
                time_28[backend].append(us / 1000000)


    time_28['group'].append(geoMean(time_28['group']))
    time_28['blas'].append(geoMean(time_28['blas']))
    time_28['mix'].append(geoMean(time_28['mix']))
    names.append('GeoMean')
    
    dat = [time_28['group'], time_28['blas'], time_28['mix']]
    apps = names
    frameworks = ['OShareMem', 'TransMM', 'Hybrid']
    num_apps = len(dat[0])
    num_bars = len(dat)

    print("[Report] mix speedup (osm, tm) a100 avg {} {} max {} {}".format(
        time_28['group'][-1]/time_28['mix'][-1],
        time_28['blas'][-1] / time_28['mix'][-1],
        max([x/y for x,y in zip(time_28['group'], time_28['mix'])]),
        max([x/y for x,y in zip(time_28['blas'], time_28['mix'])])
    ))

    
    ax = ax2
    width, lgap, ggap, gggap = 1.5, 0, 2, 0
    ind = np.arange(num_apps) * (num_bars*(width+lgap) + gggap*(num_bars//2-1) + ggap)
    for i in range(num_bars):
        ax.bar(ind + (width+lgap)*i + gggap*(i>>1),
            dat[i], width, color=color_vec[i], hatch=hatch_vec[i], edgecolor='black')
    ax.set_ylim(0, 3.5)
    ax.set_yticks(np.arange(0, 4))

    plt.setp(ax1.get_xticklabels(), fontsize=22)
    plt.setp(ax1.get_yticklabels(), fontsize=25)
    plt.setp(ax2.get_xticklabels(), fontsize=22)
    plt.setp(ax2.get_yticklabels(), fontsize=25)
    ax1.set_xticks(ind+(width+lgap+gggap/2))
    ax1.set_xticklabels(trans(apps),rotation=20)
    ax2.set_xticks(ind+(width+lgap+gggap/2))
    ax2.set_xticklabels(trans(apps),rotation=20)
    # plt.xticks(ind+(width+lgap+gggap/2), trans(apps))

    ax1.set_title('(a) A100',y=-0.3,fontsize=25)
    ax2.set_title('(b) V100',y=-0.3,fontsize=25)

    legend_handles = [mpatches.Patch(
        facecolor=color_vec[i], edgecolor='black', hatch=hatch_vec[i]) for i in range(num_bars)]
    plt.legend(legend_handles, frameworks, loc='upper center', ncol=4, bbox_to_anchor=(-0.1,1.25), fontsize=25)

    fig.subplots_adjust(wspace=0.12)

    fig.savefig(dirbase + 'two-backend.pdf', bbox_inches='tight')

def plot_multi_gpu():
    qibo_1v100 = [46.505648136138916, 0.9364736080169678, 1.6403467655181885, 19.711951732635498, 1.68607759475708, 17.76024293899536, 8.932364463806152]
    qibo_2v100 = [49.553173303604126, 7.790914535522461, 7.051988363265991, 21.656765937805176, 7.481054067611694, 18.937909364700317, 18.08937931060791]
    qibo_4v100 = [70.511310338974, 7.067523002624512, 7.14886474609375, 23.33881425857544, 6.941536903381348, 18.901276350021362, 22.0401508808136]
    my_1v100 = []
    my_2v100 = []
    my_4v100 = []
    with open(logbase + "scale.log") as f:
        for st in f.readlines():
            if "1gpu-o" in st:
                my_1v100.append(int(re.search('Time Cost: (\d+) us', st).group(1)) / 1e6)
            elif "2gpu-o" in st:
                my_2v100.append(int(re.search('Time Cost: (\d+) us', st).group(1)) / 1e6)
            elif "4gpu-o" in st:
                my_4v100.append(int(re.search('Time Cost: (\d+) us', st).group(1)) / 1e6)
    size = 8
    dat = [
        [x / y for x, y in zip(qibo_1v100, my_1v100)],
        [x / y for x, y in zip(my_1v100, my_1v100)],
        [x / y for x, y in zip(qibo_2v100, my_1v100)],
        [x / y for x, y in zip(my_2v100, my_1v100)],
        [x / y for x, y in zip(qibo_4v100, my_1v100)],
        [x / y for x, y in zip(my_4v100, my_1v100)]
    ]
    for dt in dat:
        dt.append(geoMean(dt))
    print("[Report] qibo 2v100/1v100 {} 4v100/1v100 {}".format(dat[2][-1] / dat[0][-1], dat[4][-1] / dat[0][-1]))
    print("[Report] hyquas 1v100/2v100 {} 1v100/4v100 {}".format(dat[1][-1] / dat[3][-1], dat[1][-1] / dat[5][-1]))
    print("[Report] qibo/hyquas 4v100 avg {} max {}".format(dat[4][-1]/dat[5][-1], max(x/y for x,y in zip(dat[4], dat[5]))))
    apps = ['bc', 'bv', 'hs', 'qaoa', 'qft', 'qv', 'sp', 'GeoMean']
    frameworks = ['     Qibo (1 V100)', 'HyQuas (1 V100)', '     Qibo (2 V100)', 'HyQuas (2 V100)', '     Qibo (4 V100)', 'HyQuas (4 V100)']
    num_apps = len(dat[0])
    num_bars = len(dat)

    sz = (12, 5)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)

    color_vec = [color_def[1], color_def[-1], color_def[1], color_def[-1], color_def[1], color_def[-1]]
    hatch_vec = ['', '', '..', '..', '//', '//']

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    width, lgap, ggap, gggap = 0.5, 0, 0.75, 0.2
    ind = np.arange(num_apps) * (num_bars*(width+lgap) + gggap*(num_bars//2-1) + ggap)
    for i in range(num_bars):
        ax.bar(ind + (width+lgap)*i + gggap*(i>>1),
            dat[i], width, color=color_vec[i], hatch=hatch_vec[i], edgecolor='black')
    # ax.set_ylim(0, 30)
    # ax.set_yticks(np.arange(0, 6) * 5)
    plt.yticks(fontsize=20)
    plt.ylabel('Normalized exec. time', fontsize=25)
    plt.xticks(ind+(width+lgap+gggap/2)*2.5, apps, fontsize=22)
    legend_handles = [mpatches.Patch(
        facecolor=color_vec[i], hatch=hatch_vec[i], edgecolor='black') for i in range(num_bars)]
    legend = plt.legend(legend_handles, frameworks, loc='upper center', ncol=3, fontsize='xx-large', bbox_to_anchor=(0.5, 1.3))
    # legend._legend_box.align = "right"
    #shift = max([t.get_window_extent().width for t in legend.get_texts()])
    # for t in legend.get_texts():
        # t.set_ha('right') # ha is alias for horizontalalignment
        # t.set_position((150, 0))
    
    fig.savefig(dirbase + 'v100-multi.pdf', bbox_inches='tight')

def plot_cublas():
    a100_perf = {
        '26': [],
        '27': [],
        '28': [],
        '29': [],
        '30': []
    }
    with open(logbase + "cublas-a100.log") as f:
        for n_qb in range(26, 31):
            for k in range(1, 10):
                K = f.readline()
                tm = f.readline().strip().split()
                tm = [float(t) for t in tm]
                a100_perf[str(n_qb)].append(sum(tm) * 1.0 / len(tm))
    x = list(range(1, 10))
    
    dat = [a100_perf[str(x)] for x in range(26, 31)]
    labels = ['n = ' + str(x) for x in range(26, 31)]
    num_type = len(dat)
    len_data = len(dat[0])

    x = list(range(1, 10))
    sz = (6, 4)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)

    color_vec = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
    marker_vec = marker_def

    fig, ax = plt.subplots()
    for i in range(num_type):
        ax.plot(x, dat[i], color=color_vec[i], marker=marker_vec[i], markersize=10)

    ax.set_xticks(x)
    legend_handles = [mlines.Line2D(
        [], [], color=color_vec[i], marker=marker_vec[i], label=labels[i]) for i in range(num_type)]
    plt.legend(loc='upper center', handles=legend_handles, fontsize='x-large')
    plt.xlabel('k', fontsize='xx-large')
    plt.ylabel(exectime, fontsize='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')

    plt.show()
    fig.savefig(dirbase + 'a100-cublas.pdf', bbox_inches='tight')
    
    ###
    
    v100_perf = {
        '24': [],
        '25': [],
        '26': [],
        '27': [],
        '28': []
    }
    plt.figure(figsize = (6, 4))
    with open(logbase + "cublas-v100.log") as f:
        for n_qb in range(24, 29):
            for k in range(1, 10):
                K = f.readline()
                tm = f.readline().strip().split()
                tm = [float(t) for t in tm]
                v100_perf[str(n_qb)].append(sum(tm) * 1.0 / len(tm))
    x = list(range(1, 10))
    
    dat = [v100_perf[str(x)] for x in range(24, 29)]
    labels = ['n = ' + str(x) for x in range(24, 29)]
    num_type = len(dat)
    len_data = len(dat[0])

    x = list(range(1, 10))
    sz = (6, 4)
    figsz = {'figure.figsize': sz}
    plt.rcParams.update(figsz)

    color_vec = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e']
    marker_vec = marker_def

    fig, ax = plt.subplots()
    for i in range(num_type):
        ax.plot(x, dat[i], color=color_vec[i], marker=marker_vec[i], markersize=10)

    ax.set_xticks(x)
    legend_handles = [mlines.Line2D(
        [], [], color=color_vec[i], marker=marker_vec[i], label=labels[i]) for i in range(num_type)]
    plt.legend(loc='upper center', handles=legend_handles, fontsize='x-large')
    plt.xlabel('k', fontsize='xx-large')
    plt.ylabel(exectime, fontsize='xx-large')
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')

    plt.show()
    fig.savefig(dirbase + 'v100-cublas.pdf', bbox_inches='tight')


def calc_compile():
    t_compile = []
    t_exec = []
    
    with open(logbase + "compile.log") as f:
        for s in f.readlines():
            t = re.search('-([a-z]+)/(.*?)_(\d+).log:.*? Compile Time:.*?= (\d+) us', s)
            if t is not None:
                name = t.group(2)
                backend = t.group(1)
                qubit = int(t.group(3))
                us = int(t.group(4))
                if backend == 'mix':
                    t_compile.append(us / 1000)
    
    with open(logbase + "backend.log") as f:
        for s in f.readlines():
            t = re.search('-([a-z]+)/(.*?)_(\d+).log:.*? Time Cost: (\d+) us', s)
            if t is not None:
                name = t.group(2)
                backend = t.group(1)
                qubit = int(t.group(3))
                us = int(t.group(4))
                if backend == 'mix':
                    t_exec.append(us / 1000)
    t_compile.append(geoMean(t_compile))
    t_exec.append(geoMean(t_exec))
    print("[Report] compile overhead geo:", t_compile[-1], t_exec[-1], t_compile[-1]/t_exec[-1])
    print("[Report] compile overhead (28 max): ", max([x/y for x,y in zip(t_compile, t_exec)]))
    


plot_numgate()
plot_cublas()
plot_single_gpu()
plot_multi_gpu()
plot_backend()
calc_compile()
plot_groupsz()
plot_transmm()
plot_pergate_v100()
plot_scale_v100()
plot_weak()

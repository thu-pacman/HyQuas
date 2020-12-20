import sys
import os
import numpy as np
cases = ['adder_26', 'basis_change_28', 'bv_28', 'hidden_shift_28', 'ising_25', 'qaoa_28', 'qft_28', 'quantum_volume_28', 'supremacy_28']
std_dir = sys.argv[1]
my_dir = sys.argv[2]

for case in cases:
    std = []
    with open(os.path.join(std_dir, case + '.log')) as f:
        for s in f.readlines():
            a, b = s.strip().split()[2:4]
            std.append([float(a), float(b)])
    std = np.array(std)
    std[np.abs(std) < 1e-10] = 0

    my = []
    with open(os.path.join(my_dir, case + '.log')) as f:
        for s in f.readlines():
            if s.startswith('Logger'):
                continue
            a, b = s.strip().split()[2:4]
            my.append([float(a), float(b)])
    my = np.array(my)
    my[np.abs(my) < 1e-10] = 0
    if (std.shape != my.shape):
        print("[{}]".format(case), "shape not match")
        continue
    err = np.abs(std-my)
    rela = np.abs(std - my) / (np.maximum(np.abs(std), np.abs(my)) + 1e-10)
    print("[{}]".format(case),
        "err:", np.max(err), np.argmax(err),
        "rela:", np.max(rela), np.argmax(rela))
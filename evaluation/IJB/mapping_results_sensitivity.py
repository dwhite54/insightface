#!/usr/bin/env python
# coding: utf-8
import time
import sys
import subprocess
#import traceback
#import itertools
import numpy as np
#import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import IJB_evals as IJB
#import map_tools

def get_proc_args(template, exec_root, exec_script, embs_list):
    proc_args = []
    tuple_args = []
    for is_rotation_map in [True, False]:
        for n_individuals in 2**np.arange(5, 14):
            #for n_dims in 2**np.arange(5, 8):
            for left_embs, left_dataset, left_architecture, left_head in embs_list:
                for right_embs, right_dataset, right_architecture, right_head in embs_list:
                    arg = "-s IJBC -b 64 -c 0.0001 "
                    if left_embs != right_embs:
                        arg += ' -M'

                    arg += ' -r ' + left_embs
                    arg += ' -q ' + right_embs
                    arg += ' -n ' + str(int(n_individuals))

                    save_result_name = '{}_TO_{}_N{}'.format(
                        left_embs.split('/')[-1].split('.')[0], 
                        right_embs.split('/')[-1].split('.')[0],
                        int(n_individuals),
                        #int(n_dims),
                        )

                    if is_rotation_map:
                        save_result_name += '_ortho'
                        arg += ' -R -C'
                    else:
                        save_result_name += '_ridge'

                    save_result = '../../../../results/sensitivity/{}.npz'.format(save_result_name)
                    arg += ' -S ' + save_result
                    proc_arg = template.format(cd=exec_root, script=exec_script, args=arg)
                    proc_args.append(proc_arg)
                    tuple_args.append((is_rotation_map, n_individuals, left_embs, left_dataset, left_architecture, left_head, right_embs, right_dataset, right_architecture, right_head, save_result))
    return proc_args, tuple_args

def is_machine_overloaded(machine, mem_threshold, cpu_threshold):
    proc = subprocess.Popen('ssh {machine} -T "cat /proc/loadavg && free -g && lscpu -p=cpu"'.format(machine=machine),
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    splits = proc.communicate()[0].decode('utf-8').replace('\r', '').split('\n')
    rawload = splits[0]
    rawfree = splits[2]
    n_cores = len(splits) - 1 - splits.index('0')
    load1, load5, load15 = [float(load) for load in rawload.split(' ')[:3]]
    freesplits = [int(s) for s in rawfree.split(' ') if s.isdigit()]
    available = freesplits[-1]
    total = freesplits[0]
    is_mem_overloaded = (total - available) / total > mem_threshold
    is_cpu_overloaded = load1 > n_cores*cpu_threshold or load5 > n_cores*cpu_threshold or load15 > n_cores*cpu_threshold
    return is_mem_overloaded or is_cpu_overloaded

def schedule_procs(proc_args, machines, mem_threshold=0.5, cpu_threshold=0.5, wait_time=15):
    procs = []
    i = 0
    print('Scheduling', len(proc_args), 'jobs to', len(machines), 'machines')
    #return []
    mcounts = [0]*len(machines)
    for arg in tqdm.tqdm(proc_args, file=sys.stdout):
        j = i
        while is_machine_overloaded(machines[i], mem_threshold=mem_threshold, cpu_threshold=cpu_threshold):
            #tqdm.tqdm.write(machines[i] + ' overloaded: skipping')
            i += 1
            if i >= len(machines):
                i = 0
                
            if i == j:
        machine_arg = arg.replace('MACHINE_PLACEHOLDER', machines[i])
        proc = subprocess.Popen(machine_arg, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        mcounts[i] += 1
        procs.append(proc)
        i += 1
        #print('({}/{}) began job:'.format(i, len(proc_args), machine_arg)
        if i >= len(machines):
            tqdm.tqdm.write(' '.join(['{}:{}'.format(machine, count) for machine, count in zip(machines, mcounts)]))
            i = 0
            time.sleep(wait_time)
        
    return procs

def await_procs(procs):
    i = 0
    log_fn = 'sensitivity-log.txt'
    print('Awaiting job completion, logging to {} (cancellation may leave hanging processes!)'.format(log_fn))
    with open(log_fn, 'w') as fout:
        with tqdm.tqdm(total=len(procs), file=sys.stdout) as pbar:
            while len(procs) > 0:
                poll = procs[i].poll()
                if poll is not None:
                    if poll == 0:
                        fout.write('finished job: {}\n'.format(procs[i].args))
                    else:
                        output=procs[i].communicate()
                        fout.write('error for job: {}\nSTDERR\n{}\nSTDOUT\n{}\n'.format(
                            procs[i].args, output[1].decode('utf-8'), output[0].decode('utf-8')))
                    del procs[i]
                    pbar.update(1)
                else:
                    i += 1

                if i >= len(procs):
                    time.sleep(5)
                    tqdm.tqdm.write('.', end='')
                    i = 0
                    
def gather_results(tuple_args):
    cols = np.load(tuple_args[0][-1])['FAR']
    results = np.zeros((len(tuple_args), len(cols)))
    for i in range(len(tuple_args)):
        results[i] = np.load(tuple_args[i][-1])['TAR@FAR']
        
    return results

def main():
    # TODOs: 
    #     open n_machines connections, then schedule using just those
    #     cache average cpu/mem usage per job as scheduling heuristic
    #     profile/detect machine compute power
    #     workaround max file descriptors/handles OS limit, using ? 
    
    outer_template = r'ssh MACHINE_PLACEHOLDER -T -o StrictHostKeyChecking=no "cd {cd} && python {script} {args}"'
    inner_template = r'

    embs_list = [ \
        #('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/MS1MV2-ResNet100-Arcface_IJBC.npz', 'MS1MV2', 'ResNet100', 'ArcFace'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/VGG2-ResNet50-Arcface_IJBC.npz', 'VGGFace2', 'ResNet50', 'ArcFace'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/glint360k_r100FC_0.1_IJBC.npz', 'Glint360k', 'ResNet100', 'PartialFC_r0.1'),
        #('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/glint360k_r100FC_1.0_IJBC.npz', 'Glint360k', 'ResNet100', 'PartialFC_r1.0'),
        #('/s/red/b/nobackup/data/portable/tbiom/models/arcface-tf2/ijbc_embs_arc_res50.npy', 'MS1M', 'ResNet50', 'ArcFace'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/arcface-tf2/ijbc_embs_arc_mbv2.npy', 'MS1M', 'MobileNetV2', 'ArcFace'),
        #('/s/red/b/nobackup/data/portable/tbiom/models/facenet/vggface2_ir2_ijbc_embs.npy', 'VGGFace2', 'InceptionResNetV1', 'CenterLoss'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/facenet/casia_ir2_ijbc_embs.npy', 'CASIA-WebFace', 'InceptionResNetV1', 'CenterLoss'),
        #('/s/red/b/nobackup/data/portable/tbiom/models/Probabilistic-Face-Embeddings/ijbc_embs_pfe_sphere64_msarcface_am.npy', 'MS1M', '64-CNN', 'SphereFace+PFE'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/Probabilistic-Face-Embeddings/ijbc_embs_pfe_sphere64_casia_am.npy', 'CASIA-WebFace', '64-CNN', 'SphereFace+PFE'),
    ]

    exec_root = '/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB'
    exec_script = 'IJB_evals.py'
    proc_args, tuple_args = get_proc_args(template, exec_root, exec_script, embs_list)
    
    machines = ['blue', 'cyan', 'yellow', 'magenta', 'pink', 'teal', 'aqua']
    procs = schedule_procs(proc_args, machines, mem_threshold=0.3, cpu_threshold=0.5, wait_time=60)
    
    await_procs(procs)
    
    results = gather_results(tuple_args)
    np.save('../../../../results/sensitivity/all.npy', results)
    
if __name__ == '__main__':
    main()
    
#     machines = ['blue', 'cyan', 'yellow', 'magenta', 'pink', 'teal', 'aqua']
#     for machine in machines:
#         print(machine, is_machine_overloaded(machine, 0.5, 0.5))

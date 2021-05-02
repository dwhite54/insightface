#!/usr/bin/env python
# coding: utf-8
import time
import sys
import subprocess
import traceback
import os
#import itertools
import numpy as np
#import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import IJB_evals as IJB
#import map_tools

class Machine:
    '''
    convenience class for profiling a machine and tracking how much capacity it has
    increment/decrement n_processes as you add and remove jobs
    use get_processes_available to determine how much capacity is remaining
    '''
    def __init__(self, host):
        self.host = host
        self.processes = []
        self.has_problem = False
        try:
            self.remaining_memory_MB, self._n_cpu, self._load1, self._load5, self._load15 = self._get_machine_stats(host)
            self.remaining_cpu = self._n_cpu - max(self._load1, self._load5, self._load15)
        except:
            print('Error occurred while profiling', host)
            traceback.print_exc()
            self.has_problem = True
        
    def _get_machine_stats(self, machine):
        proc = subprocess.Popen('ssh {machine} -T -o StrictHostKeyChecking=no "cat /proc/loadavg && free -m && lscpu -p=cpu"'.format(machine=machine),
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #print(proc.args)
        splits = proc.communicate()[0].decode('utf-8').replace('\r', '').split('\n')
        #print(splits)
        rawload = splits[0]
        rawfree = splits[2]
        n_cores = len(splits) - 1 - splits.index('0')
        load1, load5, load15 = [float(load) for load in rawload.split(' ')[:3]]
        available = [int(s) for s in rawfree.split(' ') if s.isdigit()][-1]
        return available, n_cores, load1, load5, load15
    
    def get_processes_available(self, memory_per_process, cpu_per_process):       
        memory_capacity_per_process = self.remaining_memory_MB / memory_per_process
        cpu_capacity_per_process = self.remaining_cpu / cpu_per_process
        memory_remaining_per_process = memory_capacity_per_process - len(self.processes)
        cpu_remaining_per_process = cpu_capacity_per_process - len(self.processes)
        return int(min(memory_remaining_per_process, cpu_remaining_per_process))
    

def schedule_procs(proc_args, hosts, log_fn, memory_per_process, cpu_per_process=1):
    '''pushes jobs to hosts as they become available, waits for them to finish, logs each jobs output to text file (log_fn)'''
    procs = []
    print('Profiling...', end='')
    print('Scheduling', len(proc_args), 'jobs to', len(hosts), 'machines, logging to', log_fn)
    print('!!!cancellation may leave open processes, use abort.py to kill these!!!')
    # get each machine's stats for capacity determination
    machines = [Machine(host) for host in hosts]
    # remove any that had issues
    machines = [machine for machine in machines if not machine.has_problem]
    print('Hosts removed due to profiling error:', len(hosts) - len(machines))
    
    total_descriptors_open = 0
    with open(log_fn, 'w') as fout:
        with tqdm.tqdm(total=len(proc_args), file=sys.stdout) as pbar:
            i, k = 0, 0  # machine index, process index
            n_complete = 0
            n_errors = 0
            while n_complete < len(proc_args):
                # resolve completed processes on machine i
                j = len(machines[i].processes) - 1
                while j >= 0:  # iterate over this machine's processes
                    poll = machines[i].processes[j].poll()
                    if poll is not None:  # process has finished
                        if poll == 0:  # process was successful
                            fout.write('finished job: {}\n'.format(machines[i].processes[j].args))
                        else:  # process failed
                            output=machines[i].processes[j].communicate()
                            fout.write('error for job: {}\nSTDERR\n{}\nSTDOUT\n{}\n'.format(
                                machines[i].processes[j].args, output[1].decode('utf-8'), output[0].decode('utf-8')))
                            n_errors += 1
                        del machines[i].processes[j]
                        n_complete += 1
                        pbar.update(1)
                    j -= 1
                
                # don't start new jobs if there are too many running (OS descriptor limit)
                if k - n_complete < 256:
                    # get machine i's remaining capacity
                    n_processes_available = machines[i].get_processes_available(memory_per_process, cpu_per_process)
                    
                    # fill remaining capacity on this machine, potentially finishing arguments
                    for _ in range(min(n_processes_available, len(proc_args) - k)):
                        process = subprocess.Popen(proc_args[k].replace('HOSTNAME', machines[i].host),
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        machines[i].processes.append(process)
                        k += 1

                # iterate to the next machine, logging after each round-robin repetition
                i += 1
                #tqdm.tqdm.write(' '.join(['k', str(k), 'n_complete', str(n_complete)]))
                if i >= len(machines):
                    #tqdm.tqdm.write(' '.join(['{}:{}'.format(machine.host, len(machine.processes)) for machine in machines]))
                    i = 0
                    #time.sleep(wait_time)
    
            print('Jobs completed:', n_complete, 'including', n_errors, 'errors')
                    

def get_proc_args():
    '''CHANGE THIS! produce a list of commands that you want to distribute across many machines'''
    embs_list = [ \
        ('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/MS1MV2-ResNet100-Arcface_IJBC.npz', 'MS1MV2', 'ResNet100', 'ArcFace'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/VGG2-ResNet50-Arcface_IJBC.npz', 'VGGFace2', 'ResNet50', 'ArcFace'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/glint360k_r100FC_0.1_IJBC.npz', 'Glint360k', 'ResNet100', 'PartialFC_r0.1'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/glint360k_r100FC_1.0_IJBC.npz', 'Glint360k', 'ResNet100', 'PartialFC_r1.0'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/arcface-tf2/ijbc_embs_arc_res50.npy', 'MS1M', 'ResNet50', 'ArcFace'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/arcface-tf2/ijbc_embs_arc_mbv2.npy', 'MS1M', 'MobileNetV2', 'ArcFace'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/facenet/vggface2_ir2_ijbc_embs.npy', 'VGGFace2', 'InceptionResNetV1', 'CenterLoss'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/facenet/casia_ir2_ijbc_embs.npy', 'CASIA-WebFace', 'InceptionResNetV1', 'CenterLoss'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/Probabilistic-Face-Embeddings/ijbc_embs_pfe_sphere64_msarcface_am.npy', 'MS1M', '64-CNN', 'SphereFace+PFE'),
        ('/s/red/b/nobackup/data/portable/tbiom/models/Probabilistic-Face-Embeddings/ijbc_embs_pfe_sphere64_casia_am.npy', 'CASIA-WebFace', '64-CNN', 'SphereFace+PFE'),
    ]
    
    template = r'ssh HOSTNAME -T -o StrictHostKeyChecking=no "cd {cd} && python {script} {args}"'
    exec_root = '/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB'
    exec_script = 'IJB_evals.py'
    
    proc_args = []
    tuple_args = []
    for left_embs, left_dataset, left_architecture, left_head in embs_list:
        for right_embs, right_dataset, right_architecture, right_head in embs_list:
            if left_embs != right_embs:
                for decay_coef in [0.0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
                    arg = "-s IJBC -b 64 -c 0.0001 -U -Q -D -M" # norm features before mapping, before eval, multiply face detector score before eval
                    if left_embs != right_embs:
                        arg += ' -r ' + left_embs
                        arg += ' -q ' + right_embs
                        arg += ' -c ' + str(decay_coef)

                        save_result_name = '{}_TO_{}_D{:.0e}'.format(
                            left_embs.split('/')[-1].split('.')[0], 
                            right_embs.split('/')[-1].split('.')[0],
                            decay_coef)

                        save_result = '../../../../results/decay/{}.npz'.format(save_result_name)
                        arg += ' -S ' + save_result
                        proc_arg = template.format(cd=exec_root, script=exec_script, args=arg)
                        proc_args.append(proc_arg)
                    #tuple_args.append((is_rotation_map, n_individuals, left_embs, left_dataset, left_architecture, left_head, right_embs, right_dataset, right_architecture, right_head, save_result))
    return proc_args, tuple_args
                    
# def gather_results(tuple_args):
#     cols = np.load(tuple_args[0][-1])['FAR']
#     results = np.zeros((len(tuple_args), len(cols)))
#     for i in range(len(tuple_args)):
#         results[i] = np.load(tuple_args[i][-1])['TAR@FAR']
        
#     return results

def main():    
    args, tuple_args = get_proc_args()
    colors = ['blue', 'cyan', 'yellow', 'magenta', 'pink', 'teal', 'aqua']
    bugs = ['ant', 'antlion', 'aphid', 'assassin-bug', 'bee', 'centipede', 'cockroach', 'cricket', 'damselfly', 'deer-fly', 'dragonfly', 'dung-beetle', 'flea', 'hornet', 'katydid', 'ladybug', 'lice', 'maggot', 'mosquito', 'moth', 'preying-mantis', 'scorpion', 'termite', 'tick', 'wasp', 'weevil']
    schedule_procs(args, colors + bugs, log_fn='dist-log.txt', memory_per_process=12000, cpu_per_process=1)
    
#     results = gather_results(tuple_args)
#     np.save('../../../../results/sensitivity/all.npy', results)
    
if __name__ == '__main__':
    main()

# TODO test product of -U, -Q, -D (false, false, false and true, true, true already complete) with n_individuals == -1

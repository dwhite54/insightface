import subprocess
machines = ['your', 'machines', 'here']  # if one of these is the current machine, put it last or the script will kill itself
procs = []
for machine in machines:
    procs.append(subprocess.Popen('ssh {machine} -T "killall --user $USER python"'.format(machine=machine),
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    
for machine, proc in zip(machines, procs):
    output = proc.communicate()
    print(machine, output[0].decode('utf-8'), output[1].decode('utf-8'))
    
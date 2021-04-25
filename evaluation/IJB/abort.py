import subprocess
colors = ['blue', 'cyan', 'yellow', 'magenta', 'pink', 'teal', 'aqua']
bugs = ['ant', 'antlion', 'aphid', 'assassin-bug', 'bee', 'centipede', 'cockroach', 'cricket', 'damselfly', 'deer-fly', 'dragonfly', 'dung-beetle', 'flea', 'hornet', 'katydid', 'ladybug', 'lice', 'maggot', 'mosquito', 'moth', 'preying-mantis', 'scorpion', 'termite', 'tick', 'wasp', 'weevil']
procs = []
machines = colors + bugs
for machine in machines:
    procs.append(subprocess.Popen('ssh {machine} -T "killall --user $USER python"'.format(machine=machine),
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    
for machine, proc in zip(machines, procs):
    output = proc.communicate()
    print(machine, output[0].decode('utf-8'), output[1].decode('utf-8'))
    
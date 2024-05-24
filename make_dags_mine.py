# 调整参数 生成仿真需要的多个DAG，在dags文件夹下

from os import system
from itertools import product

minalpha = 20
maxalpha = 50
n = 6
fat = 0.5
density = 0.5
regularity = 0.5
jump = 2

keys = ['n', 'fat', 'density', 'regularity', 'jump']
values = [n, fat, density, regularity, jump]
for i in range(10):
    param = dict(zip(keys, values))
    filename = 'dags/sim_' + str(i) + '.dot'
    param = dict(zip(keys, values))
    system("daggen-master/daggen -n {} --fat {} --density {} --regular {} --jump {} --minalpha {} --maxalpha {} --dot -o {}".format(
        param['n'],
        param['fat'],
        param['density'],
        param['regularity'],
        param['jump'],
        minalpha,
        maxalpha,
        filename
    ))
    system("dot -Tpng {} -o {}.png".format(filename, filename))
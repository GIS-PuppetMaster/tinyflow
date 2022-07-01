import datetime
import numpy as np


def get_result(path, repeat_times, net_order=0, skip=None):
    all_vdnn_vanilla_max_memory = []
    all_vdnn_vanilla_time = []

    all_capu_vanilla_max_memory = []
    all_capu_vanilla_time = []

    all_vdnn_max_memory = []
    all_vdnn_time = []
    all_vdnn_MSR = []
    all_vdnn_EOR = []
    all_vdnn_BCR = []

    all_capuchin_max_memory = []
    all_capuchin_time = []
    all_capuchin_MSR = []
    all_capuchin_EOR = []
    all_capuchin_BCR = []

    for re_t in range(repeat_times):
        res = open(f'{path}result.txt', 'w')
        if skip != 'vdnn':
            with open(f'{path}type3_repeat_time={re_t}_net_order={net_order}_record_2.txt', 'r') as f:
                lines = f.readlines()
            vanilla_max_memory = 0
            for line in lines:
                memory = float(line.split('\t')[1].split(' ')[1])
                if memory > vanilla_max_memory:
                    vanilla_max_memory = memory
            all_vdnn_vanilla_max_memory.append(vanilla_max_memory)
            with open(f'{path}type3_repeat_time={re_t}_net_order={net_order}_record_3.txt', 'r') as f:
                lines = f.readlines()
            vanilla_time = float(lines[0])
            all_vdnn_vanilla_time.append(vanilla_time)

            with open(f'{path}type2_repeat_time={re_t}_net_order={net_order}_record_2.txt', 'r') as f:
                lines = f.readlines()
            max_memory = 0
            for line in lines:
                memory = float(line.split('\t')[1].split(' ')[1])
                if memory > max_memory:
                    max_memory = memory
            all_vdnn_max_memory.append(max_memory)
            with open(f'{path}type2_repeat_time={re_t}_net_order={net_order}_record_3.txt', 'r') as f:
                lines = f.readlines()
            time = float(lines[0])
            all_vdnn_time.append(time)
            memory_saved = 1 - max_memory / vanilla_max_memory
            extra_overhead = time / vanilla_time - 1
            all_vdnn_MSR.append(memory_saved)
            all_vdnn_EOR.append(extra_overhead)
            all_vdnn_BCR.append(memory_saved / extra_overhead)
            # res.writelines('vDNN:\n')
            # res.writelines(f'max_memory:{max_memory}\n')
            # res.writelines(f'time:{time}\n')
            # res.writelines(f'memory_saved:{memory_saved}\n')
            # res.writelines(f'extra_overhead:{extra_overhead}\n')
            # res.writelines(f'efficiency:{memory_saved / extra_overhead}\n\n')

        if skip != 'capuchin':
            with open(f'{path}type0_repeat_time={re_t}_net_order={net_order}_record_2.txt', 'r') as f:
                lines = f.readlines()
            vanilla_max_memory = 0
            for line in lines:
                memory = float(line.split('\t')[1].split(' ')[1])
                if memory > vanilla_max_memory:
                    vanilla_max_memory = memory
            all_capu_vanilla_max_memory.append(vanilla_max_memory)
            with open(f'{path}type0_repeat_time={re_t}_net_order={net_order}_record_3.txt', 'r') as f:
                lines = f.readlines()
            vanilla_time = float(lines[0])
            all_capu_vanilla_time.append(vanilla_time)

            with open(f'{path}type1_repeat_time={re_t}_net_order={net_order}_record_2.txt', 'r') as f:
                lines = f.readlines()
            max_memory = 0
            for line in lines:
                try:
                    memory = float(line.split('\t')[1].split(' ')[1])
                    if memory > max_memory:
                        max_memory = memory
                except:
                    pass
            all_capuchin_max_memory.append(max_memory)
            with open(f'{path}type1_repeat_time={re_t}_net_order={net_order}_record_3.txt', 'r') as f:
                lines = f.readlines()
            time = float(lines[0])
            # try:
            #     time = float(lines[0])
            # except:
            #     print(f'exception: {path}type1_repeat_time={re_t}_net_order={net_order}_record_3.txt')
            all_capuchin_time.append(time)
            memory_saved = 1 - max_memory / vanilla_max_memory
            extra_overhead = time / vanilla_time - 1
            all_capuchin_MSR.append(memory_saved)
            all_capuchin_EOR.append(extra_overhead)
            all_capuchin_BCR.append(memory_saved / extra_overhead)


    if skip!='vdnn':
        all_vdnn_vanilla_max_memory = np.array(all_vdnn_vanilla_max_memory)
        all_vdnn_vanilla_time = np.array(all_vdnn_vanilla_time)
        all_vdnn_max_memory = np.array(all_vdnn_max_memory)
        all_vdnn_time = np.array(all_vdnn_time)
        all_vdnn_MSR = np.array(all_vdnn_MSR)
        all_vdnn_EOR = np.array(all_vdnn_EOR)
        all_vdnn_BCR = np.array(all_vdnn_BCR)
    if skip != 'capuchin':
        all_capu_vanilla_max_memory = np.array(all_capu_vanilla_max_memory)
        all_capu_vanilla_time = np.array(all_capu_vanilla_time)
        all_capuchin_max_memory = np.array(all_capuchin_max_memory)
        all_capuchin_time = np.array(all_capuchin_time)
        all_capuchin_MSR = np.array(all_capuchin_MSR)
        all_capuchin_EOR = np.array(all_capuchin_EOR)
        all_capuchin_BCR = np.array(all_capuchin_BCR)




    res.writelines('vDNN:\n')
    if skip != 'vdnn':
        res.writelines('vdnn_vanilla:\n')
        res.writelines(f'max_memory:{all_vdnn_vanilla_max_memory.mean()} +- {all_vdnn_vanilla_max_memory.std()}\n')
        res.writelines(f'time:{all_vdnn_vanilla_time.mean()} +- {all_vdnn_vanilla_time.std()}\n\n')

        res.writelines(f'max_memory:{all_vdnn_max_memory.mean()} +- {all_vdnn_max_memory.std()}\n')
        res.writelines(f'time:{all_vdnn_time.mean()} +- {all_vdnn_time.std()}\n')
        res.writelines(f'memory_saved:{all_vdnn_MSR.mean()} +- {all_vdnn_MSR.std()}\n')
        res.writelines(f'extra_overhead:{all_vdnn_EOR.mean()} +- {all_vdnn_EOR.std()}\n')
        res.writelines(f'efficiency:{all_vdnn_MSR.mean() / all_vdnn_EOR.mean()}\n\n')
    else:
        res.writelines('vdnn_vanilla:nan +- nan\n')
        res.writelines(f'max_memory:nan +- nan\n')
        res.writelines(f'time:nan +- nan\nn')

        res.writelines(f'max_memory:nan +- nan\n')
        res.writelines(f'time:nan +- nan\n')
        res.writelines(f'memory_saved:nan +- nan\n')
        res.writelines(f'extra_overhead:nan +- nan\n')
        res.writelines(f'efficiency:nan\n\n')

    res.writelines('capuchin:\n')
    if skip!='capuchin':
        res.writelines('capu_vanilla:\n')
        res.writelines(f'max_memory:{all_capu_vanilla_max_memory.mean()} +- {all_capu_vanilla_max_memory.std()}\n')
        res.writelines(f'time:{all_capu_vanilla_time.mean()} +- {all_capu_vanilla_time.std()}\n\n')

        res.writelines(f'max_memory:{all_capuchin_max_memory.mean()} +- {all_capuchin_max_memory.std()}\n')
        res.writelines(f'time:{all_capuchin_time.mean()} +- {all_capuchin_time.std()}\n')
        res.writelines(f'memory_saved:{all_capuchin_MSR.mean()} +- {all_capuchin_MSR.std()}\n')
        res.writelines(f'extra_overhead:{all_capuchin_EOR.mean()} +- {all_capuchin_EOR.std()}\n')
        res.writelines(f'efficiency:{all_capuchin_MSR.mean() / all_capuchin_EOR.mean()}\n\n')
    else:
        res.writelines('capu_vanilla:nan +- nan\n')
        res.writelines(f'max_memory:nan +- nan\n')
        res.writelines(f'time:nan +- nan\nn')

        res.writelines(f'max_memory:nan +- nan\n')
        res.writelines(f'time:nan +- nan\n')
        res.writelines(f'memory_saved:nan +- nan\n')
        res.writelines(f'extra_overhead:nan +- nan\n')
        res.writelines(f'efficiency:nan\n\n')
    res.flush()
    res.close()




def get_vanilla_max_memory(path, repeat_times, net_order=0):
    all_vanilla_max_memory = []
    for re_t in range(repeat_times):
        try:
            with open(f'{path}type0_repeat_time={re_t}_net_order={net_order}_record_2.txt', 'r') as f:
                lines = f.readlines()
        except:
            break
        vanilla_max_memory = 0
        for line in lines:
            try:
                memory = float(line.split('\t')[1].split(' ')[1])
                if memory > vanilla_max_memory:
                    vanilla_max_memory = memory
            except:
                pass
        all_vanilla_max_memory.append(vanilla_max_memory)
    all_vanilla_max_memory = np.array(all_vanilla_max_memory)
    return all_vanilla_max_memory.mean()


if __name__ == '__main__':
    get_result('log/VGG/', repeat_times=1, need_tosave=[5770.907183725366, 5771.339125520944, 6711.676414494596])

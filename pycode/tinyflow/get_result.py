import numpy as np
import traceback
import os

def get_result(raw_workload, repeat_times):
    all_saved_ratio = []
    all_saved_ratio_cold_start = []
    all_extra_overhead = []
    all_vanilla_max_memory_used = []
    all_schedule_max_memory_used = []
    all_schedule_max_memory_used_cold_start = []
    all_vanilla_time_cost = []
    all_schedule_time_cost = []
    all_memory_saved_to_extra_overhead_ratio = []
    for i in range(repeat_times):
        workload = raw_workload + f'/repeat_{i}'
        vanilla_path = f'{workload}/vanilla/'
        scheduled_path = f'{workload}/schedule/'
        with open(vanilla_path + f'gpu_record_cold_start.txt', 'r') as f:
            lines = f.readlines()
        try:
            temp = lines[-1].split('\t')
            vanilla_max_memory_used = float(temp[2].split(' ')[1])
        except:
            temp = lines[-2].split('\t')
            vanilla_max_memory_used = float(temp[2].split(' ')[1])
        all_vanilla_max_memory_used.append(vanilla_max_memory_used)
        with open(scheduled_path + f'gpu_record.txt', 'r') as f:
            lines = f.readlines()
        try:
            temp = lines[-1].split('\t')
            schedule_max_memory_used = float(temp[2].split(' ')[1])
        except:
            temp = lines[-2].split('\t')
            schedule_max_memory_used = float(temp[2].split(' ')[1])
        all_schedule_max_memory_used.append(schedule_max_memory_used)
        saved_ratio = 1 - schedule_max_memory_used / vanilla_max_memory_used
        all_saved_ratio.append(saved_ratio)

        with open(scheduled_path + f'gpu_record_cold_start.txt', 'r') as f:
            lines = f.readlines()
        try:
            temp = lines[-1].split('\t')
            schedule_max_memory_used = float(temp[2].split(' ')[1])
        except:
            temp = lines[-2].split('\t')
            schedule_max_memory_used = float(temp[2].split(' ')[1])
        all_schedule_max_memory_used_cold_start.append(schedule_max_memory_used)
        saved_ratio = 1 - schedule_max_memory_used / vanilla_max_memory_used
        all_saved_ratio_cold_start.append(saved_ratio)

        with open(vanilla_path + 'gpu_time.txt', 'r') as f:
            lines = f.readlines()
        vanilla_time_cost = float(lines[0].replace('time_cost:', ''))
        all_vanilla_time_cost.append(vanilla_time_cost)
        with open(scheduled_path + 'gpu_time.txt', 'r') as f:
            lines = f.readlines()
        schedule_time_cost = float(lines[0].replace('time_cost:', ''))
        all_schedule_time_cost.append(schedule_time_cost)
        extra_overhead = schedule_time_cost/vanilla_time_cost - 1
        all_extra_overhead.append(extra_overhead)
        memory_saved_to_extra_overhead_ratio = saved_ratio / extra_overhead
        all_memory_saved_to_extra_overhead_ratio.append(memory_saved_to_extra_overhead_ratio)
    all_saved_ratio = np.array(all_saved_ratio)
    all_saved_ratio_cold_start = np.array(all_saved_ratio_cold_start)
    all_extra_overhead = np.array(all_extra_overhead)
    all_vanilla_max_memory_used = np.array(all_vanilla_max_memory_used)
    all_schedule_max_memory_used = np.array(all_schedule_max_memory_used)
    all_schedule_max_memory_used_cold_start = np.array(all_schedule_max_memory_used_cold_start)
    all_vanilla_time_cost = np.array(all_vanilla_time_cost)
    all_schedule_time_cost = np.array(all_schedule_time_cost)
    # all_memory_saved_to_extra_overhead_ratio = np.array(all_memory_saved_to_extra_overhead_ratio)

    with open(f'{raw_workload}/repeat_{repeat_times}_result.txt', 'w') as f:
        f.write(f'saved_ratio:{all_saved_ratio.mean()} +- {all_saved_ratio.std()}'
                f'\nextra_overhead:{all_extra_overhead.mean()} +- {all_extra_overhead.std()}'
                f'\nvanilla_max_memory_used:{all_vanilla_max_memory_used.mean()} +- {all_vanilla_max_memory_used.std()}'
                f'\nschedule_max_memory_used:{all_schedule_max_memory_used.mean()} +- {all_schedule_max_memory_used.std()}'
                f'\nvanilla_time_cost:{all_vanilla_time_cost.mean()} +- {all_vanilla_time_cost.std()}'
                f'\nschedule_time_cost:{all_schedule_time_cost.mean()} +- {all_schedule_time_cost.std()}'
                f'\nefficiency:{all_saved_ratio.mean() / all_extra_overhead.mean()}'
                f'\n\n\nsaved_ratio_cold_start:{all_saved_ratio_cold_start.mean()} +- {all_saved_ratio_cold_start.std()}'
                f'\nschedule_max_memory_used_cold_start:{all_schedule_max_memory_used_cold_start.mean()} +- {all_schedule_max_memory_used_cold_start.std()}'
                f'\nefficiency_cold_start:{all_saved_ratio_cold_start.mean() / all_extra_overhead.mean()}')


# if __name__ == '__main__':
#     from MakeCSV import filelist
#     for p in filelist:
#         try:
#             get_result(p, 3)
#             print(f'完成{p}')
#         except Exception as e:
#             print(f'跳过path:{p}')
#             traceback.print_exc()

#Examples of using functions from the package nqcpresonator - analysis module
import os
from nqcpresonator import analysis

#Describes a sample containing multiple resonators
subject_id = "B002019D02"
#Directory for power sweep experiments (Deep, Low, High, Quick overall and reproducibility)
d_data = r'C:\Users\dfz956\2024-08-26 W4_Al_resonators4_B002019D02 1\2024-08-26 W4_Al_resonators4_B002019D02\2024-08-26 W4_Al_resonators4_B002019D02-run2\data' 
#Change directory to find power sweep data files
os.chdir(d_data) 

def Protocol1():
    #Number of resonators
    no_resonators = 12

    # List of file names
    file_lists = [
        [f'LowPowerSweep_B002019D02_{i}.csv' for i in range(no_resonators)],
        [f'HighPowerSweep_B002019D02_{i}.csv' for i in range(no_resonators)]
    ]
    power_shifts = [0, 0]

    analysis.presentResults(file_lists,power_shifts,no_resonators)

    file_lists = [
        [f'RepSweep_B002019D02_20dB(-70to-60_)0_{i}.csv' for i in range(no_resonators)], 
        [f'DeepPowerSweep_20dB_{i}.csv' for i in range(no_resonators)],
        [f'LowPowerSweep_B002019D02_{i}.csv' for i in range(no_resonators)],
        [f'HighPowerSweep_B002019D02_{i}.csv' for i in range(no_resonators)]
    ]
    power_shifts = [-20, -20, 0, 0]

    #analysis.presentResults(file_lists,power_shifts,no_resonators)

    file_lists = [
        [f'RepSweep(-70to-80_)0_{i}.csv' for i in range(12)],
        [f'RepSweep(-70to-80_)1_{i}.csv' for i in range(12)],
        [f'RepSweep(-70to-80_)2_{i}.csv' for i in range(12)]
    ]
    power_shifts = [0, 0, 0]

    analysis.presentResults(file_lists,power_shifts,no_resonators)

#These functions are bugged 
def Protocol2():
    no_resonators = 12
    design_f_r = [5e9, 5.1e9, 5.2e9, 5.3e9, 6e9, 6.1e9, 6.2e9, 6.3e9, 7e9, 7.1e9, 7.2e9, 7.3e9]

    # List of file names for each dataset
    # CHANGE OVERNIGHTDEEP TO WHATEVER SWEEP PROTOCOL FILE NAME HAS BEEN USED
    # CHANGE range(8) IF THERE ARE NOT 8 RESONATORS
    datasets = {
        'sweep1': [f'RepSweep(-70to-80_)0_{i}.csv' for i in range(no_resonators)],
        'sweep2': [f'RepSweep(-70to-80_)1_{i}.csv' for i in range(no_resonators)],
        'sweep3': [f'RepSweep(-70to-80_)2_{i}.csv' for i in range(no_resonators)]
    }

    analysis.presentMOI(datasets,design_f_r,no_resonators,subject_id)
    
Protocol2()
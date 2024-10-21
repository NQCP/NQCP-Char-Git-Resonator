from __future__ import division, absolute_import, print_function
#matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from resonator import background, see, shunt
from resonator.shunt import LinearShuntFitter
import pandas as pd
from resonator.background import MagnitudeSlopeOffsetPhaseDelay 

#A power sweep is represented by a csv file with the columns: freq, power, mag, phase, re, im
#"freq" is the measured frequency and "power" is the applied power
#"mag","phase" and "re","im" represent the transmission parameter (s_21) in polar and rectangular form respectively

#The MOI are resonant frequency (f_r) internal quality factor (Q_i) and the coupling quality factor (Q_c)

#Protocol 1 : Power Sweeps
###########################################################################################################

def load_csv_data(file_name):
    '''Converts a csv file to a Pandas data frame'''
    df = pd.read_csv(file_name)
    return df

def conv(mag, phase):
    '''Converts a complex number from polar to rectangular representation'''
    mag_linear = 10**(mag / 20)
    s21_real = mag_linear * np.cos(np.deg2rad(phase))
    s21_imag = mag_linear * np.sin(np.deg2rad(phase))
    s21 = s21_real + 1j * s21_imag
    return s21

#def shiftPower(df,power_shift)
#Shifts the power in the dataframe, df, representing a power sweep

def organize_data(file_lists, power_shifts = []):
    '''Each file in "file_lists" represents a power sweep. This function concatenates them into one large data set and returns the columns (freq, power, mag, phase, re, im)'''
    #Is the power shift argument optional?
    num_resonators = len(file_lists[0])
    num_points = 4000

    # Initialize the lists to store data
    freqs_array = []
    mags_array = []
    phases_array = []
    s21_array = []
    unique_powers = []

    for resonator_idx in range(num_resonators):
        freqs_array_resonator = []
        mags_array_resonator = []
        phases_array_resonator = []
        s21_array_resonator = []
        powers_resonator = []

        for file_list_idx, file_list in enumerate(file_lists):
            file_name = file_list[resonator_idx]
            data = load_csv_data(file_name)
            #Power is changed discretly, so much the array elements are duplicate
            #Here the unique values are extracted, and the power shift added.
            powers = data['pow'].unique() + power_shifts[file_list_idx]  # Apply power shift here
            powers_resonator.extend(powers) #The array of unique, shifted powers is added to a list with the remaining files
            
            for power in powers:
                #For each power level (single element in array), there are many elements for the freq,mag,phase and s21.
                #Here these are extracted
                subset = data[data['pow'] == power - power_shifts[file_list_idx]]  # Adjust subset selection
                freqs_array_resonator.append(subset['freq'].values)
                mags_array_resonator.append(subset['mag'].values)
                phases_array_resonator.append(subset['phase'].values)
                s21_array_resonator.append(conv(subset['mag'].values, subset['phase'].values))

        #These are addded to a large array with the remaining files
        freqs_array.append(freqs_array_resonator)
        mags_array.append(mags_array_resonator)
        phases_array.append(phases_array_resonator)
        s21_array.append(s21_array_resonator)
        unique_powers.append(np.array(powers_resonator))

    return freqs_array, mags_array, phases_array, s21_array, unique_powers
    
def fit_resonator(frequency, data):
    '''The input to this function is the measured frequency and the transmission parameter (in rectangular representation).
    These are fitted using a Daniel Flanigans model that assumes a hanger mode configuration and system response (background) for which the magnitude varies linearly with frequency and there is a
    fixed time delay and phase offset.
    The output is the resonant frequencies, internal and coupling quality factors.'''
    try:
        resonator = LinearShuntFitter(frequency=frequency, data=data,
                                      background_model=MagnitudeSlopeOffsetPhaseDelay())
        if any(value is None for value in [resonator.f_r, resonator.Q_i, resonator.Q_i_error, resonator.Q_c, resonator.Q_c_error]):
            print("Fit returned None for some values. Returning None.")
            return None, None, None, None, None
        return resonator.f_r, resonator.Q_i, resonator.Q_i_error, resonator.Q_c, resonator.Q_c_error
    except Exception as e:
        print(f"Error fitting data: {e}")
        return None, None, None, None, None

def fit_and_save_results(file_lists, power_shifts):
    '''Applies the function "fit_resonator" to each resonator in the array "file_lists".
       The results (f_r,Q_i, Q_c) are saved in a csv file.'''
    for file_list_idx, file_list in enumerate(file_lists):
        results = []
        for resonator_idx, file_name in enumerate(file_list):
            data = load_csv_data(file_name)
            powers = data['pow'].unique() + power_shifts[file_list_idx]  # Apply power shift here
            
            for power in powers:
                subset = data[data['pow'] == power - power_shifts[file_list_idx]]  # Adjust subset selection
                frequency = subset['freq'].values
                s21_data = conv(subset['mag'].values, subset['phase'].values)
                f_r, Q_i, Q_i_err, Q_c, Q_c_err = fit_resonator(frequency, s21_data)
                
                # Skip the results if the fitting was not successful
                if None in [f_r, Q_i, Q_i_err, Q_c, Q_c_err]:
                    print(f"Skipping resonator {resonator_idx} at power {power} due to unsuccessful fit.")
                    continue
                
                results.append([resonator_idx, power, f_r, Q_i, Q_i_err, Q_c, Q_c_err])
        
        if results:  # Save results only if there are valid entries
            df = pd.DataFrame(results, columns=['Resonator Index', 'Power (dBm)', 'f_r (Hz)', 'Q_i', 'Q_i_error', 'Q_c', 'Q_c_error'])
            sweep_name = file_lists[file_list_idx][0].split('.')[0].replace('0', '').rstrip('0123456789')
            df.to_csv(f'MOIs_{sweep_name}.csv', index=False)
        else:
            print(f"No valid results for file list {file_list_idx}. No CSV file generated.")

def fit_all_resonators(freqs_array, s21_array, unique_powers,no_resonators):
    '''The input to this function is a set of power sweeps collected into a single data structure (see "organize_data").
        The output are the power levels, resonant frequencies, internal and coupling quality factors.'''
    all_results = []
    for resonator_idx in range(no_resonators):  # Adjust number of resonators as needed
        results = []
        for power_idx in range(len(unique_powers[resonator_idx])):
            frequency = freqs_array[resonator_idx][power_idx]
            data = s21_array[resonator_idx][power_idx]
            f_r, Q_i, Q_i_err, Q_c, Q_c_err = fit_resonator(frequency, data)
            
            # Skip the results if the fitting was not successful
            if None in [f_r, Q_i, Q_i_err, Q_c, Q_c_err]:
                print(f"Skipping resonator {resonator_idx} at power index {power_idx} due to unsuccessful fit.")
                continue
            
            results.append([unique_powers[resonator_idx][power_idx], f_r, Q_i, Q_i_err, Q_c, Q_c_err])
        
        if results:
            power_levels, f_rs, Q_is, Qi_errors, Q_cs, Qc_errors = zip(*results)
            all_results.append((power_levels, f_rs, Q_is, Qi_errors, Q_cs, Qc_errors))
        else:
            print(f"No valid results for resonator {resonator_idx}.")
    
    return all_results

def plot_individual_resonator_results(all_resonator_results):
    '''For a resonator, this function plots internal and coupling quality factors vs. power'''
    for idx, (power_levels, f_rs, Q_is, Qi_errors, Q_cs, Qc_errors) in enumerate(all_resonator_results):
        if len(f_rs) == 0 or len(Q_is) == 0 or len(Q_cs) == 0:
            print(f"No data to plot for resonator {idx+1}. Skipping plot.")
            continue
        
        plt.figure(figsize=(10, 6))
        
        # Plot Qi
        plt.errorbar(power_levels, Q_is, yerr=Qi_errors, marker='o', label=f'Qi', color='b', capsize=5, ls="none")
        
        # Plot Qc
        plt.errorbar(power_levels, Q_cs, yerr=Qc_errors, marker='o', label=f'Qc', color='r', capsize=5, ls="none")
        
        plt.xlabel('Power (dBm)')
        plt.ylabel('Q')
        plt.ylim([0, max(max(Q_is), max(Q_cs)) * 1.1])
        plt.title(f'Resonator {idx+1} - Qi and Qc vs Power')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(f'Resonator_{idx+1}_Qi_Qc_vs_Power.png')
        
        # Show plot
        plt.show()

def plot_results(all_resonator_results):
    '''Plots Q_i and Q_c vs. power for each resonator.'''
    plt.figure(figsize=(12, 8))
    for idx, (power_levels, f_rs, Q_is, Qi_errors, Q_cs, Qc_errors) in enumerate(all_resonator_results):
        if len(Q_is) == 0:
            print(f"No Qi data to plot for resonator {idx+1}. Skipping plot.")
            continue
        
        plt.errorbar(power_levels, Q_is, yerr=Qi_errors, marker='o', label=f'Resonator {idx+1}', capsize=5)
    
    plt.xlabel('Power (dBm)')
    plt.ylabel('Qi')
    plt.ylim([0, 2e7])
    plt.title('Qi vs Power for each Resonator')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 8))
    for idx, (power_levels, f_rs, Q_is, Qi_errors, Q_cs, Qc_errors) in enumerate(all_resonator_results):
        if len(Q_cs) == 0:
            print(f"No Qc data to plot for resonator {idx+1}. Skipping plot.")
            continue
        
        plt.errorbar(power_levels, Q_cs, yerr=Qc_errors, marker='o', label=f'Resonator {idx+1}', capsize=5)
    
    plt.xlabel('Power (dBm)')
    plt.ylabel('Qc')
    plt.ylim([0, 8e5])
    plt.title('Qc vs Power for each Resonator')
    plt.legend()
    plt.grid(True)
    plt.show()

# Constants
hbar = 1.054571817e-34  # Reduced Planck's constant (in Joules*seconds)
Z0 = 50  # Impedance of the system (Ohms), assuming a 50 Ohm system, can be adjusted
Zr = 50  # Resonator impedance (Ohms), can be adjusted

def calculate_Q_tot(Q_i, Q_c):
    '''Returns the total quality factor Q_tot'''
    return 1 / (1 / Q_i + 1 / Q_c)

def calculate_photon_number(Q_tot, Qc, fr, power_dBm):
    '''Converts the power in dBm to a average number of photons for a specific Q_tot, Q_c and f_r'''
    # Convert power from dBm to Watts
    power_watts = 10**((power_dBm-73) / 10) / 1000  # P in Watts
    
    # Convert resonant frequency to angular frequency (rad/s)
    omega0 = 2 * np.pi * fr
    
    # Calculate <n> using the updated formula with Q_tot
    n_avg = ((2/(hbar*omega0**2))*((Q_tot**2)/(Qc)))*power_watts
    
    return n_avg


def add_secondary_photon_number_axis(ax, power_levels, photon_numbers):
    '''Auxillary function for plotting'''
    # Create a secondary x-axis on top of the current plot
    ax_photon = ax.twiny()

    ax_photon.set_xlim(ax.get_xlim()) 

    # Manually choose fewer tick positions (e.g., every 3rd point)
    tick_positions = power_levels[::3] 
    tick_labels = photon_numbers[::3]  

    # Format the labels conditionally
    formatted_labels = [f'{n:.2e}' if n >= 1000 else f'{n:.2f}' for n in tick_labels]

    ax_photon.set_xticks(tick_positions)
    ax_photon.set_xticklabels(formatted_labels, rotation=90)

    ax_photon.set_xlabel('Average Photon Number <n>', fontsize=12, labelpad=15)


    return ax_photon

def plot_individual_resonator_results_with_photons(all_resonator_results):
    '''Plots Q_i and Q_c versus power (in dBM and photons) for each resonator.'''
    max_value = 5e6  # Maximum allowed value for Q factors and their errors

    for idx, (power_levels, f_rs, Q_is, Qi_errors, Q_cs, Qc_errors) in enumerate(all_resonator_results):
        if len(f_rs) == 0 or len(Q_is) == 0 or len(Q_cs) == 0:
            print(f"No data to plot for resonator {idx+1}. Skipping plot.")
            continue

        # Filter out points where Qi, Qc, or their errors exceed the threshold
        filtered_results = [
            (pwr, fr, Qi, Qi_err, Qc, Qc_err)
            for pwr, fr, Qi, Qi_err, Qc, Qc_err in zip(power_levels, f_rs, Q_is, Qi_errors, Q_cs, Qc_errors)
            if (Qi <= max_value and Qi_err <= max_value and Qc <= max_value and Qc_err <= max_value)
        ]

        # If no valid data points remain after filtering, skip this resonator
        if len(filtered_results) == 0:
            print(f"No valid data to plot for resonator {idx+1} after filtering. Skipping plot.")
            continue

        # Unzip the filtered results back into individual lists
        power_levels_filtered, f_rs_filtered, Q_is_filtered, Qi_errors_filtered, Q_cs_filtered, Qc_errors_filtered = zip(*filtered_results)

        # Calculate Q_tot for each filtered data point
        Q_tot_list = [calculate_Q_tot(Q_is_filtered[i], Q_cs_filtered[i]) for i in range(len(Q_is_filtered))]

        # Calculate average photon numbers for each filtered power level
        photon_numbers = [calculate_photon_number(Q_tot_list[i], Q_cs_filtered[i], f_rs_filtered[i], power_levels_filtered[i]) for i in range(len(power_levels_filtered))]


        # Plot Qi vs Power (dBm)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(power_levels_filtered, Q_is_filtered, yerr=Qi_errors_filtered, 
                    marker='o', label=f'Qi', color='#1f77b4', markersize=8, 
                    markerfacecolor='white', capsize=5, ls="none", lw=1.5)
        
        ax.errorbar(power_levels_filtered, Q_cs_filtered, yerr=Qc_errors_filtered, 
                    marker='s', label=f'Qc', color='#ff7f0e', markersize=8, 
                    markerfacecolor='white', capsize=5, ls="none", lw=1.5)
        
        ax.set_xlabel('Power (dBm)', fontsize=12)
        ax.set_ylabel('Quality Factor (Q)', fontsize=12)
        ax.set_ylim([0, max(max(Q_is_filtered), max(Q_cs_filtered)) * 1.1])
        ax.set_title(f'Resonator {idx+1} - Qi and Qc vs Power', fontsize=14, pad=20)

        ax.legend(fontsize=10, loc='best')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add secondary x-axis for average photon number
        ax_photon = add_secondary_photon_number_axis(ax, power_levels_filtered, photon_numbers)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'Resonator_{idx+1}_Qi_Qc_vs_Power_and_photons_singlephoton.png')

        # Show the plot
        plt.show()
        print(Q_tot_list)
        print(Q_cs_filtered)
        print(power_levels_filtered)
        print(f_rs_filtered)
        print(photon_numbers)

def presentResults(file_lists,power_shifts,no_resonators):
    '''Fits, saves and plots results for each resonator in arg. file_list with power shift in arg. power shift'''
    #no_resonators can be replaced by the length of the array, file_lists
    freqs_array, mags_array, phases_array, s21_array, unique_powers = organize_data(file_lists, power_shifts)
    fit_and_save_results(file_lists, power_shifts)
    all_resonator_results = fit_all_resonators(freqs_array,s21_array, unique_powers,no_resonators)
    #plot_results(all_resonator_results)
    plot_individual_resonator_results_with_photons(all_resonator_results)



#Protocol 2: Low Power Reproducibility
###########################################################################################################

def plot(name, subject_id, design_f_r):
    final_csv_file = name
    combined_results_df = pd.read_csv(final_csv_file)

    # Extract data for plotting
    resonators = combined_results_df['resonator_idx'] + 1  # Adjusting index for plotting (1-12)
    f_r_avgs = combined_results_df['f_r_avg']
    f_r_errors = combined_results_df['f_r_err']
    Q_i_avgs = combined_results_df['Q_i_avg']
    Q_i_errors = combined_results_df['Q_i_err']
    Q_c_avgs = combined_results_df['Q_c_avg']
    Q_c_errors = combined_results_df['Q_c_err']

    # Calculate the Qi/Qc ratio and its error propagation
    combined_results_df['Qi_Qc_ratio'] = combined_results_df['Q_i_avg'] / combined_results_df['Q_c_avg']
    combined_results_df['Qi_Qc_err'] = combined_results_df['Qi_Qc_ratio'] * np.sqrt(
        (combined_results_df['Q_i_err'] / combined_results_df['Q_i_avg'])**2 +
        (combined_results_df['Q_c_err'] / combined_results_df['Q_c_avg'])**2
    )

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))  # Adjust the figure size as needed
    fig.suptitle(f'Resonator Analysis {subject_id}: f_r, Q_i, Q_c, and Qi/Qc Ratio', fontsize=24)  # Increased font size

    # Plot 1: Resonance Frequency (f_r)
    axs[0, 0].errorbar(resonators, f_r_avgs, yerr=f_r_errors, fmt='o', label='f_r_avg',
                    capsize=5, color='darkgreen', ecolor='darkgreen', elinewidth=2,
                    marker='D', markersize=8, markerfacecolor='darkgreen', alpha=0.6)
    
    axs[0, 0].plot(resonators, design_f_r, 'ro', label='Design f_r', linewidth=2) 

    axs[0, 0].set_xlabel('Resonator idx', fontsize=16)  # Increased font size
    axs[0, 0].set_ylabel('Resonance Frequency (f_r)', fontsize=16)  # Increased font size
    axs[0, 0].set_title('Resonance Frequency (f_r)', fontsize=18)  # Increased font size
    axs[0, 0].set_xticks(resonators)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
    axs[0, 0].legend(fontsize=14)  # Increased legend font size
    axs[0, 0].grid(True)

    # Plot 2: Intrinsic Quality Factor (Q_i)
    axs[0, 1].errorbar(resonators, Q_i_avgs, yerr=Q_i_errors, fmt='o', label='Q_i_avg',
                    capsize=5, color='blue', ecolor='blue', elinewidth=2,
                    marker='o', markersize=8, markerfacecolor='blue', alpha=0.6)
    axs[0, 1].set_xlabel('Resonator idx', fontsize=16)  # Increased font size
    axs[0, 1].set_ylabel('Intrinsic Quality Factor (Q_i)', fontsize=16)  # Increased font size
    axs[0, 1].set_title('Intrinsic Quality Factor (Q_i)', fontsize=18)  # Increased font size
    axs[0, 1].set_xticks(resonators)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
    axs[0, 1].grid(True)

    # Plot 3: Coupling Quality Factor (Q_c)
    axs[1, 0].errorbar(resonators, Q_c_avgs, yerr=Q_c_errors, fmt='o', label='Q_c_avg',
                    capsize=5, color='orange', ecolor='orange', elinewidth=2,
                    marker='s', markersize=8, markerfacecolor='orange', alpha=0.6)
    axs[1, 0].set_xlabel('Resonator idx', fontsize=16)  # Increased font size
    axs[1, 0].set_ylabel('Coupling Quality Factor (Q_c)', fontsize=16)  # Increased font size
    axs[1, 0].set_title('Coupling Quality Factor (Q_c)', fontsize=18)  # Increased font size
    axs[1, 0].set_xticks(resonators)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
    axs[1, 0].grid(True)

    # Plot 4: Qi / Qc Ratio
    axs[1, 1].errorbar(resonators, combined_results_df['Qi_Qc_ratio'], yerr=combined_results_df['Qi_Qc_err'], fmt='o',
                    label='Qi/Qc ratio', capsize=5, color='purple', ecolor='purple',
                    elinewidth=2, marker='^', markersize=8, markerfacecolor='purple', alpha=0.6)
    axs[1, 1].set_xlabel('Resonator idx', fontsize=16)  # Increased font size
    axs[1, 1].set_ylabel('Qi / Qc Ratio', fontsize=16)  # Increased font size
    axs[1, 1].set_title('Intrinsic to Coupling Quality Factor Ratio (Qi/Qc)', fontsize=18)  # Increased font size
    axs[1, 1].set_xticks(resonators)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
    axs[1, 1].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the master figure
    plt.savefig('master_figure_resonator_analysis.png', dpi=400)
    plt.show()

def weighted_average(values, errors):
    weights = 1 / np.square(errors)
    weighted_avg = np.sum(values * weights) / np.sum(weights)
    weighted_error = np.sqrt(1 / np.sum(weights))
    return weighted_avg, weighted_error

def save_results_to_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)

def presentMOI(datasets,design_f_r,no_resonators,subject_id):
    '''Iterates through each sweep in the array datasets and finds MOIs to be exported to a csv file and plots them'''
    for name, file_names in datasets.items():
        #print(file_names)
        freqs_array, mags_array, phases_array, s21_array = organize_data(file_names)
        results = fit_all_resonators(freqs_array, s21_array)
        save_results_to_csv(results, f'{name}_results.csv')

    combined_results = []

    for resonator_idx in range(no_resonators):
        all_f_rs = []
        all_f_r_errors = []
        all_Q_is = []
        all_Q_i_errors = []
        all_Q_cs = []
        all_Q_c_errors = []
        
        for name in datasets.keys():
            df = load_csv_data(f'{name}_results.csv')
            all_f_rs.append(df.loc[resonator_idx, 'f_r_avg'])
            all_f_r_errors.append(df.loc[resonator_idx, 'f_r_err'])
            all_Q_is.append(df.loc[resonator_idx, 'Q_i_avg'])
            all_Q_i_errors.append(df.loc[resonator_idx, 'Q_i_err'])
            all_Q_cs.append(df.loc[resonator_idx, 'Q_c_avg'])
            all_Q_c_errors.append(df.loc[resonator_idx, 'Q_c_err'])
        
        all_f_rs = np.array(all_f_rs)
        all_f_r_errors = np.array(all_f_r_errors)
        all_Q_is = np.array(all_Q_is)
        all_Q_i_errors = np.array(all_Q_i_errors)
        all_Q_cs = np.array(all_Q_cs)
        all_Q_c_errors = np.array(all_Q_c_errors)
        
        f_r_avg, f_r_err = weighted_average(all_f_rs, all_f_r_errors)
        Q_i_avg, Q_i_err = weighted_average(all_Q_is, all_Q_i_errors)
        Q_c_avg, Q_c_err = weighted_average(all_Q_cs, all_Q_c_errors)
        
        combined_results.append({
            'resonator_idx': resonator_idx,
            'f_r_avg': f_r_avg, 'f_r_err': f_r_err,
            'Q_i_avg': Q_i_avg, 'Q_i_err': Q_i_err,
            'Q_c_avg': Q_c_avg, 'Q_c_err': Q_c_err
        })

    save_results_to_csv(combined_results, 'combined_results.csv')
    file_name='combined_results.csv'
    plot(file_name, subject_id, design_f_r)
    
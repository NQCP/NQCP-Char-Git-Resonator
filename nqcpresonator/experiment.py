import pyvisa
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

#...

#file handling definitions. just execute

#General resonator measurement protocol
##########################################################################################################################################
def VNAStart():
    subject_id = "B002019D02"
    #plug in and turn on VNA. Open software "Network Analyzer" and click run

    rm = pyvisa.ResourceManager()
    inst = rm.open_resource('TCPIP0::SCI1025305::hislip_PXI0_CHASSIS1_SLOT1_INDEX0::INSTR')

    inst.timeout = 300000000
    inst.write("SYST:ERR?")
    print(inst.read())
    print(inst.query("*IDN?"))
    inst.write("CALC:PAR:SEL 'CH1_S11_1'")
    inst.write("FORM:DATA ASCII")
    inst.write("CALC:PAR:MOD S21")
    inst.write("INIT1:CONT OFF")

def save_to_csv(data, index, filename_prefix):
    filename = f'{filename_prefix}{index}.csv'
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['freq', 'pow', 'mag', 'phase', 're', 'im'])
        for row in data:
            writer.writerow(row)

def save_parameters_to_csv(filename, num_averages, if_bandwidth, num_points, span, pmin, pmax, pstep):
    with open(f'{filename}_PCPs.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Average', 'IF Bandwidth', 'Number of Points', 'Span', 'Pmin', 'Pmax', 'Pstep'])
        writer.writerow([num_averages, if_bandwidth, num_points, span, pmin, pmax, pstep])

#Protocol 0 : Overview
##########################################################################################################################################
def Overview(inst, num_averages, start_freq, stop_freq, num_points, if_bandwidth, power_level):
    inst.write(f"SENS1:AVER:COUN {num_averages}")
    inst.write(f"SENS1:FREQ:STAR {start_freq}")
    inst.write(f"SENS1:FREQ:STOP {stop_freq}")
    inst.write(f"SENS:SWE:POIN {num_points}")
    inst.write(f"SENS1:BAND {if_bandwidth}")
    inst.write(f"SOUR1:POW {power_level}")

    inst.write("INIT1:CONT OFF")
    inst.write("SENS1:AVER ON")
    inst.write("INIT1:IMM;*WAI")

    inst.write("CALC1:DATA? SDATA")
    data = inst.read()

    data = data.split(',')
    data = list(map(float, data))  
    real = data[0::2]
    imag = data[1::2]
    complex_data = np.array(real) + 1j * np.array(imag)

    frequency = np.linspace(start_freq, stop_freq, num_points)
    phase = np.degrees(np.angle(complex_data)) 

    plt.figure(figsize=(20, 6))
    plt.plot(frequency, 20 * np.log10(np.abs(complex_data)), label='Magnitude')
    plt.title(f'VNA Sweep Result {subject_id}, Power level = {power_level}dBm')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend()
    plt.savefig("Overview/Overview_magnitude.png", dpi = 300)
    plt.show()


    plt.figure(figsize=(20, 6))
    plt.plot(frequency, phase, label='Phase')
    plt.title(f'VNA Sweep Result {subject_id}, Power level = {power_level}dBm')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase(deg)')
    plt.grid(True)
    plt.legend()
    plt.savefig("Overview/Overview_phase.png", dpi = 300)
    plt.show()

# def estimateResonance()

# Lorentzian function definition
def lorentzian(f, f0, gamma, a, b):
    return a * (1 - (gamma**2) / ((f - f0)**2 + gamma**2)) + b

def inspectResonance():
    num_averages = 1
    num_points = 1000
    if_bandwidth = 100
    power_level = 0
    span =1.5e6
    
    # Initialize dictionaries to store data for each resonator
    singres_magnitude = {}
    singres_phase = {}
    singres_complex = {}
    singres_real = {}
    singres_imaginary = {}
    singres_frequency = {}
    resonance_frequencies = []
    fwhm_values = []

    inst.write(f"SENS:SWE:POIN {num_points}")
    inst.write(f"SENS1:BAND {if_bandwidth}")
    inst.write(f"SOUR1:POW {power_level}")

    for index, center_freq in enumerate(reslist):
        inst.write(f"SENS1:AVER:COUN {num_averages}")
        inst.write(f"SENS1:FREQ:CENT {center_freq}")
        inst.write(f"SENS1:FREQ:SPAN {span}")

        inst.write("INIT1:CONT OFF")
        inst.write("SENS1:AVER ON")
        inst.write("INIT1:IMM;*WAI")

        time.sleep(1)
        
        inst.write("CALC1:DATA? SDATA")
        data = inst.read()

        data = data.split(',')
        data = list(map(float, data))
        real = data[0::2]
        imag = data[1::2]
        complex_data = np.array(real) + 1j*np.array(imag)
        magnitude = 20 * np.log10(np.abs(complex_data))
        phase = np.angle(complex_data, deg=True)

        frequency = np.linspace(center_freq - span / 2, center_freq + span / 2, num_points)

        singres_frequency[index+1] = frequency
        singres_magnitude[index+1] = magnitude
        singres_phase[index+1] = phase
        singres_complex[index+1] = complex_data
        singres_real[index+1] = real
        singres_imaginary[index+1] = imag

        # Fit the magnitude data to the Lorentzian function
        popt, _ = curve_fit(lorentzian, frequency, magnitude, 
                            p0=[center_freq, 3e5, np.min(magnitude), np.max(magnitude)])

        f0, gamma, a, b = popt
        fwhm = 2 * gamma

        # Store the resonance frequency and FWHM
        resonance_frequencies.append(f0)
        fwhm_values.append(fwhm)

        # Plotting magnitude vs. frequency with the fit
        plt.figure(figsize=(10, 6))
        plt.plot(frequency, magnitude, label='Data')
        plt.plot(frequency, lorentzian(frequency, *popt), label=f'Fit: f0={f0/1e9:.3f} GHz, FWHM={fwhm/1e6:.3f} MHz')
        plt.title(f'Resonator{index+1}: VNA Sweep Result at Center Frequency {center_freq/1e9:.2f} GHz')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'Overview/Resonator{index+1} mag.png')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(frequency, phase, label=f'Center {center_freq/1e9} GHz')
        plt.title(f'Resonator{index+1}: VNA Sweep Result at Center Frequency {center_freq/1e9:.2f} GHz')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (deg)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'Overview/Resonator{index+1} phase.png')
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(real, imag, label=f'Center {center_freq/1e9} GHz')
        plt.title(f'Resonator{index+1}: VNA Sweep Result at Center Frequency {center_freq/1e9:.2f} GHz')
        plt.xlabel('Re(S21)')
        plt.ylabel('Im(S21)')
        plt.grid(True)
        plt.legend()
        plt.savefig(f'Overview/Resonator{index+1} Im vs Re.png')
        plt.show()

    # At the end of the loop, you can print or store the resonance frequencies and FWHM values
    print("Resonance Frequencies (Hz):", resonance_frequencies)
    print("FWHM (Hz):", fwhm_values)

# Protocol 1 : Power Sweep
##########################################################################################################################################

def measure_resonators(inst, reslist, num_averages, num_points, if_bandwidth, pmin, pmax, pstep, span, filename):
    power_levels = np.linspace(pmin, pmax, pstep)
    resonator_data = {}
    all_data = {}

    inst.write("CALC:PAR:SEL 'CH1_S11_1'")
    inst.write("FORM:DATA ASCII")
    inst.write("CALC:PAR:MOD S21")
    
    inst.write(f"SENS1:AVER:COUN {num_averages}")
    inst.write(f"SENS:SWE:POIN {num_points}")
    inst.write(f"SENS1:BAND {if_bandwidth}")

    for index, center_freq in enumerate(reslist): 
        resind = [] 
        magnitude_data = []

        for power_level in power_levels:
            inst.write(f"SOUR1:POW {power_level}")
            inst.write(f"SENS1:FREQ:CENT {center_freq}")
            inst.write(f"SENS1:FREQ:SPAN {span}")

            inst.write("INIT1:CONT OFF")
            inst.write("SENS1:AVER ON")
            inst.write("INIT1:IMM;*WAI")

            time.sleep(1)  # Waiting for measurement to stabilize

            inst.write("CALC1:DATA? SDATA")
            data = inst.read()

            data = data.split(',')
            data = list(map(float, data))
            real = data[0::2]
            imag = data[1::2]
            complex_data = np.array(real) + 1j * np.array(imag)
            magnitude_db = 20 * np.log10(np.abs(complex_data))
            phase_deg = np.degrees(np.angle(complex_data))

            frequency_range = np.linspace(center_freq - span / 2, center_freq + span / 2, num_points)

            for freq, mag, phase, re, im in zip(frequency_range, magnitude_db, phase_deg, real, imag):
                resind.append([freq, power_level, mag, phase, re, im])

            magnitude_data.append(magnitude_db)

        all_data[index] = np.array(resind)
        save_to_csv(resind, index, filename) 

        magnitude_data = np.array(magnitude_data)

        plt.figure(figsize=(12, 8))
        plt.imshow(magnitude_data, aspect='auto', 
                   extent=[center_freq - span / 2, center_freq + span / 2, power_levels[-1], power_levels[0]], 
                   cmap='RdBu', interpolation='nearest')
        plt.colorbar(label='Magnitude (dB)')
        plt.title(f'Resonator{index+1}: VNA Sweep Result at Center Frequency {center_freq/1e9:.2f} GHz')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Level (dBm)')
        plt.ylim([pmin, pmax])
        plt.savefig(f'PowerSweeps/Resonator{index+1}_{filename}')
        plt.show()

    save_parameters_to_csv(filename, num_averages, if_bandwidth, num_points, span, pmin, pmax, pstep)

    return all_data

def measure_resonators_span(inst, reslist, num_averages, num_points, if_bandwidth, pmin, pmax, pstep, fwhm_list, filename):
    power_levels = np.linspace(pmin, pmax, pstep)
    resonator_data = {}
    all_data = {}

    inst.write("CALC:PAR:SEL 'CH1_S11_1'")
    inst.write("FORM:DATA ASCII")
    inst.write("CALC:PAR:MOD S21")
    
    inst.write(f"SENS1:AVER:COUN {num_averages}")
    inst.write(f"SENS:SWE:POIN {num_points}")
    inst.write(f"SENS1:BAND {if_bandwidth}")

    for index, (center_freq, fwhm) in enumerate(zip(reslist, fwhm_list)): 
        resind = [] 
        magnitude_data = []

        # Calculate span as 10 times the FWHM
        span = 10 * fwhm

        # Debugging print to check if span is calculated correctly
        print(f"Resonator {index+1}: Center Frequency = {center_freq}, FWHM = {fwhm}, Span = {span}")

        # Ensure the span is non-zero and positive
        if span <= 0:
            print(f"Error: Calculated span is {span}. Skipping this resonator.")
            continue

        for power_level in power_levels:
            inst.write(f"SOUR1:POW {power_level}")
            inst.write(f"SENS1:FREQ:CENT {center_freq}")
            inst.write(f"SENS1:FREQ:SPAN {span}")

            inst.write("INIT1:CONT OFF")
            inst.write("SENS1:AVER ON")
            inst.write("INIT1:IMM;*WAI")

            time.sleep(1)  # Waiting for measurement to stabilize

            inst.write("CALC1:DATA? SDATA")
            data = inst.read()

            data = data.split(',')
            data = list(map(float, data))
            real = data[0::2]
            imag = data[1::2]
            complex_data = np.array(real) + 1j * np.array(imag)
            magnitude_db = 20 * np.log10(np.abs(complex_data))
            phase_deg = np.degrees(np.angle(complex_data))

            frequency_range = np.linspace(center_freq - span / 2, center_freq + span / 2, num_points)

            for freq, mag, phase, re, im in zip(frequency_range, magnitude_db, phase_deg, real, imag):
                resind.append([freq, power_level, mag, phase, re, im])

            magnitude_data.append(magnitude_db)

        all_data[index] = np.array(resind)
        save_to_csv(resind, index, filename) 

        magnitude_data = np.array(magnitude_data)

        plt.figure(figsize=(12, 8))
        plt.imshow(magnitude_data, aspect='auto', 
                   extent=[center_freq - span / 2, center_freq + span / 2, power_levels[-1], power_levels[0]], 
                   cmap='RdBu', interpolation='nearest')
        plt.colorbar(label='Magnitude (dB)')
        plt.title(f'Resonator{index+1}: VNA Sweep Result at Center Frequency {center_freq/1e9:.2f} GHz')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Level (dBm)')
        plt.ylim([pmin, pmax])
        plt.savefig(f'PowerSweeps/Resonator{index+1}_{filename}')
        plt.show()

    # Save parameters for the last used span, could be generalized if needed
    save_parameters_to_csv(filename, num_averages, if_bandwidth, num_points, span, pmin, pmax, pstep)

    return all_data

# Protocol 2 : Reproducbility
##########################################################################################################################################
def LP_rep(inst, reslist, num_averages, num_points, if_bandwidth, pmin, pmax, pstep, fwhm_list, num_repeats, filename):
    power_levels = np.linspace(pmin, pmax, pstep)
    resonator_data = {}
    all_data = {}

    inst.write("CALC:PAR:SEL 'CH1_S11_1'")
    inst.write("FORM:DATA ASCII")
    inst.write("CALC:PAR:MOD S21")
    
    inst.write(f"SENS1:AVER:COUN {num_averages}")
    inst.write(f"SENS:SWE:POIN {num_points}")
    inst.write(f"SENS1:BAND {if_bandwidth}")

    for repeat in range(num_repeats):
        for index, (center_freq, fwhm) in enumerate(zip(reslist, fwhm_list)):
            resind = []
            magnitude_data = []

            # Calculate span as 10 times the FWHM
            span = 10 * fwhm

            # Debugging print to check if span is calculated correctly
            print(f"Repeat {repeat + 1}, Resonator {index + 1}: Center Frequency = {center_freq}, FWHM = {fwhm}, Span = {span}")

            for power_level in power_levels:
                inst.write(f"SOUR1:POW {power_level}")
                inst.write(f"SENS1:FREQ:CENT {center_freq}")
                inst.write(f"SENS1:FREQ:SPAN {span}")

                inst.write("INIT1:CONT OFF")
                inst.write("SENS1:AVER ON")
                inst.write("INIT1:IMM;*WAI")

                time.sleep(1)  # Waiting for measurement to stabilize

                inst.write("CALC1:DATA? SDATA")
                data = inst.read()

                data = data.split(',')
                data = list(map(float, data))
                real = data[0::2]
                imag = data[1::2]
                complex_data = np.array(real) + 1j * np.array(imag)
                magnitude_db = 20 * np.log10(np.abs(complex_data))
                phase_deg = np.degrees(np.angle(complex_data))

                frequency_range = np.linspace(center_freq - span / 2, center_freq + span / 2, num_points)

                for freq, mag, phase, re, im in zip(frequency_range, magnitude_db, phase_deg, real, imag):
                    resind.append([freq, power_level, mag, phase, re, im])

                magnitude_data.append(magnitude_db)

            all_data[index] = np.array(resind)
            save_to_csv(resind, f'{repeat}_{index}', filename)

            magnitude_data = np.array(magnitude_data)

            plt.figure(figsize=(12, 8))
            plt.imshow(magnitude_data, aspect='auto', 
                       extent=[center_freq - span / 2, center_freq + span / 2, power_levels[-1], power_levels[0]], 
                       cmap='RdBu', interpolation='nearest')
            plt.colorbar(label='Magnitude (dB)')
            plt.title(f'Resonator{index+1}:Power Sweep Heatmap at Center Frequency {center_freq / 1e9:.2f} GHz')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Level (dBm)')
            plt.ylim([pmin, pmax])
            plt.savefig(f"Reproducibility/Resonator{index+1}_rep{repeat+1}_{filename}")
            plt.show()

    # Save parameters for the last used span (can be generalized if needed)
    save_parameters_to_csv(filename, num_averages, if_bandwidth, num_points, span, pmin, pmax, pstep, num_repeats)

    return all_data
import numpy as np
import os
import sys
import datetime
import re
from matplotlib import pyplot as plt
import h5py
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.cluster import KMeans

def create_folder_if_not_exists(folder_path):
    import os
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def datetime_to_unix(dt):
    # Convert to Unix timestamp
    unix_timestamp = int(dt.timestamp())
    return unix_timestamp

def unix_to_datetime(unix_timestamp):
    # Convert the Unix timestamp to a datetime object
    dt = datetime.fromtimestamp(unix_timestamp)
    return dt

def load_from_h5(filename, data_type, save_r=1):  # Added save_r as parameter.

    data = {data_type: {}}  # Initialize the main dictionary with the data_type.

    with h5py.File(filename, 'r') as f:
        for qubit_group in f.keys():
            qubit_index = int(qubit_group[1:]) - 1
            qubit_data = {}
            group = f[qubit_group]

            for dataset_name in group.keys():
                # Attempt to map HDF5 keys to the target dictionaries' keys.
                if data_type == 'Res':
                    target_keys = {'Dates': 'Dates', 'freq_pts': 'freq_pts', 'freq_center': 'freq_center',
                                       'Amps': 'Amps', 'Found Freqs': 'Found Freqs', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'QSpec':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Frequencies': 'Frequencies',
                                       'I Fit': 'I Fit', 'Q Fit': 'Q Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'Ext_QSpec':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Frequencies': 'Frequencies','Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num', 'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}

                elif data_type == 'Rabi':
                    target_keys = {'Dates': 'Dates', 'I': 'I', 'Q': 'Q', 'Gains': 'Gains', 'Fit': 'Fit',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'SS':
                    target_keys = {'Fidelity': 'Fidelity', 'Angle': 'Angle', 'Dates': 'Dates', 'I_g': 'I_g',
                                       'Q_g': 'Q_g', 'I_e': 'I_e', 'Q_e': 'Q_e',
                                       'Round Num': 'Round Num', 'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'T1':
                    target_keys = {'T1': 'T1', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'T2':
                    target_keys = {'T2': 'T2', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'T2E':
                    target_keys = {'T2E': 'T2E', 'Errors': 'Errors', 'Dates': 'Dates', 'I': 'I', 'Q': 'Q',
                                       'Delay Times': 'Delay Times', 'Fit': 'Fit', 'Round Num': 'Round Num',
                                       'Batch Num': 'Batch Num','Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type == 'stark2D':
                    target_keys = {'Dates': 'Dates', 'I':'I', 'Q': 'Q', 'Qu Frequency Sweep':'Qu Frequency Sweep',
                                   'Res Gain Sweep':'Res Gain Sweep','Round Num':'Round Num', 'Batch Num': 'Batch Num',
                                   'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}
                elif data_type =='starkSpec':
                    target_keys = {'Dates': 'Dates', 'I':'I', 'Q': 'Q','P': 'P', 'shots':'shots','Gain Sweep':'Gain Sweep','Round Num':'Round Num', 'Batch Num': 'Batch Num',
                                   'Exp Config': 'Exp Config', 'Syst Config': 'Syst Config'}

                else:
                        raise ValueError(f"Unsupported data_type: {data_type}")

                try:
                    mapped_key = target_keys[dataset_name]  # Map HDF5 key to target key.
                    qubit_data[mapped_key] = [group[dataset_name][()]] * save_r  # Expand to match the desired length.

                except KeyError:
                    print(f"Warning: Key '{dataset_name}' not found in target dictionary for data_type '{data_type}'. Skipping.")
                    pass

            data[data_type][qubit_index] = qubit_data

    return data

def process_string_of_nested_lists(data):
    # Remove extra whitespace and non-numeric characters.
    data = re.sub(r'\s*\[(\s*.*?\s*)\]\s*', r'[\1]', data)
    data = data.replace('[ ', '[')
    data = data.replace('[ ', '[')
    data = data.replace('[ ', '[')
    cleaned_data = ''.join(c for c in data if c.isdigit() or c in ['-', '.', ' ', 'e', '[', ']'])
    pattern = r'\[(.*?)\]'  # Regular expression to match data within brackets
    matches = re.findall(pattern, cleaned_data)
    result = []
    for match in matches:
        numbers = [float(x.strip('[').strip(']').replace("'", "").replace(" ", "").replace("  ", "")) for x in
                    match.split()]  # Convert strings to integers
    result.append(numbers)

    return result

def process_h5_data(data):
    # Check if the data is a byte string; decode if necessary.
    if isinstance(data, bytes):
        data_str = data.decode()
    elif isinstance(data, str):
        data_str = data
    else:
        raise ValueError("Unsupported data type. Data should be bytes or string.")

    # Remove extra whitespace and non-numeric characters.
    cleaned_data = ''.join(c for c in data_str if c.isdigit() or c in ['-', '.', ' ', 'e'])

    # Split into individual numbers, removing empty strings.
    numbers = [float(x) for x in cleaned_data.split() if x]
    return numbers

def get_abs_min(start_time, dates):
#returns absolute time in minutes
    abs_min = []
    for date in dates:
        abs_min.append(np.array((date - start_time).total_seconds()) / 60)
    return abs_min

class qspec:
    def __init__(self, data_dir, dataset, QubitIndex, folder="study_data", expt_name="qspec_ge"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.expt_name = expt_name
        self.folder = folder

    def fit_qspec(self, I, Q, freqs):
        mag = np.sqrt(np.square(I) + np.square(Q))
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(mag)]
        qfreq, qfreq_err, fwhm, qspec_fit = self.fit_lorenzian(mag, freqs, freq_q)
        return qfreq, qfreq_err, fwhm, qspec_fit

    def lorentzian(self,f, f0, gamma, A, B):
        return A * gamma ** 2 / ((f - f0) ** 2 + gamma ** 2) + B

    def max_offset_difference_with_x(self,x_values, y_values, offset):
        max_average_difference = -1
        corresponding_x = None

        # average all 3 to avoid noise spikes
        for i in range(len(y_values) - 2):
            # group 3 vals
            y_triplet = y_values[i:i + 3]

            # avg differences for these 3 vals
            average_difference = sum(abs(y - offset) for y in y_triplet) / 3

            # see if this is the highest difference yet
            if average_difference > max_average_difference:
                max_average_difference = average_difference
                # x value for the middle y value in the 3 vals
                corresponding_x = x_values[i + 1]

        return corresponding_x, max_average_difference

    def fit_lorenzian(self, mag, freqs, freq_q):
            # Initial guesses for I and Q
            initial_guess = [freq_q, 1, np.max(mag), np.min(mag)]

            # First round of fits (to get rough estimates)
            params, _ = curve_fit(self.lorentzian, freqs, mag, p0=initial_guess)


            # Use these fits to refine guesses
            x_max_diff, max_diff = self.max_offset_difference_with_x(freqs, mag, params[3])
            initial_guess = [x_max_diff, 1, np.max(mag), np.min(mag)]

            # Second (refined) round of fits, this time capturing the covariance matrices
            params, cov = curve_fit(self.lorentzian, freqs, mag, p0=initial_guess)

            # Create the fitted curves
            fit = self.lorentzian(freqs, *params)

            # Calculate errors from the covariance matrices
            fit_err = np.sqrt(np.diag(cov))[0]

            # Extract fitted means and FWHM (assuming params[0] is the mean and params[1] relates to the width)
            mean = params[0]
            fwhm = 2 * params[1]

            # Calculate the amplitude differences from the fitted curves
            amp_fit = abs(np.max(fit) - np.min(fit))

            return mean, fit_err, fwhm, fit

    def load_all(self):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files = os.listdir(data_path)
        h5_files.sort()
        n = len(h5_files)

        dates = []
        I = []
        Q = []

        load_data = load_from_h5(os.path.join(data_path, h5_files[0]), 'QSpec', save_r=1)
        qspec_probe_freqs = process_h5_data(load_data['QSpec'][self.QubitIndex].get('Frequencies', [])[0][0].decode())

        for h5_file in h5_files:
            load_data = load_from_h5(os.path.join(data_path, h5_file), 'QSpec', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['QSpec'][self.QubitIndex].get('Dates', [])[0][0]))

            I.append(np.array(process_h5_data(load_data['QSpec'][self.QubitIndex].get('I', [])[0][0].decode())))
            Q.append(np.array(process_h5_data(load_data['QSpec'][self.QubitIndex].get('Q', [])[0][0].decode())))

        return dates, n, qspec_probe_freqs, I, Q

    def get_all_qspec_freq(self, qspec_probe_freqs, I, Q, n):
        qspec_freqs = []
        qspec_errs = []
        fwhms = []

        for round in np.arange(0,n):
            qfreq, qfreq_err, fwhm, qspec_fit = self.fit_qspec(I[round], Q[round], qspec_probe_freqs)
            qspec_freqs.append(qfreq)
            qspec_errs.append(qfreq_err)
            fwhms.append(fwhm)

        return qspec_freqs, qspec_errs, fwhms

    def get_qspec_freq_in_round(self, qspec_probe_freqs, I, Q, round, n, plot=False):
        thisI = I[round]
        thisQ = Q[round]

        qfreq, qfreq_err, fwhm, qspec_fit = self.fit_qspec(thisI, thisQ, qspec_probe_freqs)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(qspec_probe_freqs, np.sqrt(np.square(thisI) + np.square(thisQ)), label='data')
            ax.plot(qspec_probe_freqs, qspec_fit, label='lorentzian')
            ax.legend()
            ax.set_xlabel('qubit probe frequency [MHz]')
            ax.set_ylabel('I,Q magnitude [a.u.]')
            ax.set_title(f'dataset {self.dataset} qubit {self.QubitIndex + 1} round {round + 1} of {n} low-gain qspec: {np.round(qfreq,2)} +/- {np.round(qfreq_err,2)} MHz, fwhm: {np.round(fwhm,2)} MHz')
            plt.show()

        return qfreq, qfreq_err, fwhm, qspec_fit

class t1:

    def __init__(self,data_dir, dataset, QubitIndex, theta, threshold, folder="study_data", expt_name ="t1_ge", thresholding = True):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.expt_name = expt_name
        self.folder = folder
        self.theta = theta
        self.threshold = threshold
        self.thresholding = thresholding

    def load_all(self):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files = os.listdir(data_path)
        h5_files.sort()
        n = len(h5_files)

        dates = []
        I_shots = []
        Q_shots = []

        load_data = load_from_h5(os.path.join(data_path, h5_files[0]), 'T1', save_r=1)
        delay_times = process_h5_data(load_data['T1'][self.QubitIndex].get('Delay Times', [])[0][0].decode())
        steps = len(delay_times)
        reps = int(len(process_h5_data(load_data['T1'][self.QubitIndex].get('I', [])[0][0].decode()))/steps)

        for h5_file in h5_files:
            load_data = load_from_h5(os.path.join(data_path, h5_file), 'T1', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['T1'][self.QubitIndex].get('Dates', [])[0][0]))

            I_shots.append(np.array(process_h5_data(load_data['T1'][self.QubitIndex].get('I', [])[0][0].decode())).reshape([reps, steps]))
            Q_shots.append(np.array(process_h5_data(load_data['T1'][self.QubitIndex].get('Q', [])[0][0].decode())).reshape([reps, steps]))

        return dates, n, delay_times, steps, reps, I_shots, Q_shots

    def plot_shots(self, I_shots, Q_shots, delay_times, n, round=0, idx=10):
        print(np.shape(I_shots))

        this_I = I_shots[round][:,idx]
        this_Q = Q_shots[round][:,idx]

        print(np.shape(this_I))

        i_new = this_I * np.cos(self.theta) - this_Q * np.sin(self.theta)
        q_new = this_I * np.sin(self.theta) + this_Q * np.cos(self.theta)

        states = (i_new > self.threshold)

        fig, ax = plt.subplots()
        ax.scatter(i_new, q_new, c=states)
        ax.set_xlabel('I [a.u.]')
        ax.set_ylabel('Q [a.u.]')
        ax.set_title(f'dataset {self.dataset} qubit {self.QubitIndex} round {round + 1} of {n}: rotated I,Q shots for t1_ge at delay time: {np.round(delay_times[idx],2)} us')
        plt.show()

    def process_shots(self, I_shots, Q_shots, n, steps):

        p_excited = []
        for round in np.arange(n):
            p_excited_in_round = []
            for idx in np.arange(steps):
                this_I = I_shots[round][:, idx]
                this_Q = Q_shots[round][:, idx]

                i_new = this_I * np.cos(self.theta) - this_Q * np.sin(self.theta)
                q_new = this_I * np.sin(self.theta) + this_Q * np.cos(self.theta)
                if self.thresholding:
                    states = (i_new > self.threshold)
                else:
                    states = np.mean(i_new)
                p_excited_in_round.append(np.mean(states))

            p_excited.append(p_excited_in_round)

        return p_excited

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def t1_fit(self, signal, delay_times, round, n, plot=False):

        # Initial guess for parameters
        q1_a_guess = np.max(signal) - np.min(signal)  # Initial guess for amplitude (a)
        q1_b_guess = 0  # Initial guess for time shift (b)
        q1_c_guess = (delay_times[-1] - delay_times[0]) / 5  # Initial guess for decay constant (T1)
        q1_d_guess = np.min(signal)  # Initial guess for baseline (d)

        # Form the guess array
        q1_guess = [q1_a_guess, q1_b_guess, q1_c_guess, q1_d_guess]

        # Define bounds to constrain T1 (c) to be positive, but allow amplitude (a) to be negative
        lower_bounds = [-np.inf, -np.inf, 0, -np.inf]  # Amplitude (a) can be negative/positive, but T1 (c) > 0
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]  # No upper bound on parameters

        # Perform the fit using the 'trf' method with bounds
        q1_popt, q1_pcov = curve_fit(self.exponential, delay_times, signal,
                                     p0=q1_guess, bounds=(lower_bounds, upper_bounds),
                                     method='trf', maxfev=10000)

        # Generate the fitted exponential curve
        q1_fit_exponential = self.exponential(delay_times, *q1_popt)

        # Extract T1 and its error
        T1_est = q1_popt[2]  # Decay constant T1
        T1_err = np.sqrt(q1_pcov[2][2]) if q1_pcov[2][2] >= 0 else float('inf')  # Ensure error is valid

        if plot:
            fig, ax = plt.subplots()
            ax.plot(delay_times, signal, label='data')
            ax.plot(delay_times, q1_fit_exponential, label='exponential')
            ax.set_xlabel('Delay Time [us]')
            ax.set_ylabel('P(e)')
            ax.legend()
            ax.set_title(f'dataset {self.dataset} qubit {self.QubitIndex + 1} round {round + 1} of {n}: t1_ge = {T1_est:.3f} +/- {T1_err} us')
            plt.show()


        return q1_fit_exponential, T1_err, T1_est

    def get_all_t1(self, delay_times, p_excited, n):

        t1s = []
        t1_errs = []

        for round in np.arange(n):
            p_excited_in_round = p_excited[round][:]
            q1_fit_exponential, T1_err, T1_est = self.t1_fit(p_excited_in_round, delay_times, round, n, plot=False)
            t1s.append(T1_est)
            t1_errs.append(T1_err)

        return t1s, t1_errs

    def get_t1_in_round(self, delay_times, p_excited, n, round, plot=True):
        p_excited_in_round = p_excited[round][:]
        q1_fit_exponential, T1_err, T1_est = self.t1_fit(p_excited_in_round, delay_times, round, n, plot=plot)

        return q1_fit_exponential, T1_err, T1_est

class ssf:
    def __init__(self,data_dir, dataset, QubitIndex, folder="study_data", expt_name ="ss_ge"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.expt_name = expt_name
        self.folder = folder

    def load_all(self):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files = os.listdir(data_path)
        h5_files.sort()
        n = len(h5_files)

        dates = []
        I_g = []
        Q_g = []
        I_e = []
        Q_e = []
        fid = []
        theta = []

        for h5_file in h5_files:
            load_data = load_from_h5(os.path.join(data_path, h5_file), 'SS', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['SS'][self.QubitIndex].get('Dates', [])[0][0]))

            I_g.append(process_h5_data(load_data['SS'][self.QubitIndex].get('I_g', [])[0][0].decode()))
            Q_g.append(process_h5_data(load_data['SS'][self.QubitIndex].get('Q_g', [])[0][0].decode()))
            I_e.append(process_h5_data(load_data['SS'][self.QubitIndex].get('I_e', [])[0][0].decode()))
            Q_e.append(process_h5_data(load_data['SS'][self.QubitIndex].get('Q_e', [])[0][0].decode()))
            fid.append(load_data['SS'][self.QubitIndex].get('Fidelity', [])[0])
            theta.append(load_data['SS'][self.QubitIndex].get('Angle', [])[0])

        return dates, n, I_g, Q_g, I_e, Q_e, fid, theta

    def get_ssf_in_round(self, I_g, Q_g, I_e, Q_e, round):
        ig = np.array(I_g[round])
        qg = np.array(Q_g[round])
        ie = np.array(I_e[round])
        qe = np.array(Q_e[round])

        xg, yg = np.median(ig), np.median(qg)
        xe, ye = np.median(ie), np.median(qe)

        """Compute the rotation angle"""
        theta = -np.arctan2((ye - yg), (xe - xg))
        """Rotate the IQ data"""
        ig_new = ig * np.cos(theta) - qg * np.sin(theta)
        qg_new = ig * np.sin(theta) + qg * np.cos(theta)
        ie_new = ie * np.cos(theta) - qe * np.sin(theta)
        qe_new = ie * np.sin(theta) + qe * np.cos(theta)

        """New means of each blob"""
        xg, yg = np.median(ig_new), np.median(qg_new)
        xe, ye = np.median(ie_new), np.median(qe_new)

        return ig_new, qg_new, ie_new, qe_new, xg, yg, xe, ye

class resstarkspec:

    def __init__(self, data_dir, dataset, QubitIndex, stark_constant, theta, threshold, folder = "study_data", expt_name = "res_starkspec_ge", thresholding=True):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.stark_constant = stark_constant
        self.folder = folder
        self.expt_name = expt_name
        self.theta = theta
        self.threshold = threshold
        self.thresholding = thresholding

    def load_all(self):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files = os.listdir(data_path)
        h5_files.sort()
        n = len(h5_files)

        dates = []
        I_shots = []
        Q_shots = []
        P = []

        load_data = load_from_h5(os.path.join(data_path, h5_files[0]), 'starkSpec', save_r=1)
        gain_sweep = process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Gain Sweep', [])[0][0].decode())
        steps = len(gain_sweep)
        reps = int(len(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('I', [])[0][0].decode())) / steps)

        for h5_file in h5_files:
            load_data = load_from_h5(os.path.join(data_path, h5_file), 'starkSpec', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['starkSpec'][self.QubitIndex].get('Dates', [])[0][0]))

            I_shots.append(np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('I', [])[0][0].decode())).reshape(
                    [reps, steps]))
            Q_shots.append(np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Q', [])[0][0].decode())).reshape(
                    [reps, steps]))
            P.append(np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('P', [])[0][0].decode())))

        return dates, n, gain_sweep, steps, reps, I_shots, Q_shots, P

    def plot_shots(self, I_shots, Q_shots, gains, n, round=0, idx=10):

        this_I = I_shots[round][idx,:]
        this_Q = Q_shots[round][idx,:]

        i_new = this_I * np.cos(self.theta) - this_Q * np.sin(self.theta)
        q_new = this_I * np.sin(self.theta) + this_Q * np.cos(self.theta)

        states = (i_new > self.threshold)

        fig, ax = plt.subplots()
        ax.scatter(i_new, q_new, c=states)
        ax.set_xlabel('I [a.u.]')
        ax.set_ylabel('Q [a.u.]')
        ax.set_title(f'dataset {self.dataset} qubit {self.QubitIndex +1} round {round + 1} of {n}: rotated I,Q shots for res_stark_spec at gain: {np.round(gains[idx],2)} us')
        plt.show()

    def process_shots(self, I_shots, Q_shots, n, steps):

        p_excited = []
        for round in np.arange(n):
            p_excited_in_round = []
            for idx in np.arange(steps):
                this_I = I_shots[round][:,idx]
                this_Q = Q_shots[round][:,idx]

                i_new = this_I * np.cos(self.theta) - this_Q * np.sin(self.theta)
                q_new = this_I * np.sin(self.theta) + this_Q * np.cos(self.theta)
                if self.thresholding:
                    states = (i_new > self.threshold)
                else:
                    states = np.mean(i_new)
                p_excited_in_round.append(np.mean(states))

            p_excited.append(p_excited_in_round)

        return p_excited

    def gain2freq(self, gains):
        freqs = np.square(gains) * self.stark_constant
        return freqs

    def get_p_excited_in_round(self, gains, p_excited, n, round, plot=True):
        p_excited_in_round = p_excited[round]

        if plot:
            fig, ax = plt.subplots(2,1, layout='constrained')
            fig.suptitle(f'dataset {self.dataset} qubit {self.QubitIndex + 1} round {round + 1} of {n} resonator stark spectroscopy')

            ax[0].plot(gains, p_excited_in_round)
            ax[0].set_xlabel('resonator stark gain [a.u.]')
            ax[0].set_ylabel('P(e)')

            ax[1].plot(self.gain2freq(gains), p_excited_in_round)
            ax[1].set_xlabel('stark shift [MHz]')
            ax[1].set_ylabel('P(e)')
            plt.show()

        return p_excited_in_round

class starkspec:

    def __init__(self, data_dir, dataset, QubitIndex, duffing_constant, theta, threshold, anharmonicity, detuning, folder = "study_data", expt_name = "starkspec_ge", thresholding=True):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.duffing_constant = duffing_constant
        self.folder = folder
        self.expt_name = expt_name
        self.theta = theta
        self.threshold = threshold
        self.thresholding = thresholding
        self.anharmonicity = anharmonicity
        self.detuning = detuning

    def load_all(self):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files = os.listdir(data_path)
        h5_files.sort()
        n = len(h5_files)

        dates = []
        I_shots = []
        Q_shots = []
        P = []

        load_data = load_from_h5(os.path.join(data_path, h5_files[0]), 'starkSpec', save_r=1)
        gain_sweep = process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Gain Sweep', [])[0][0].decode())
        steps = len(gain_sweep)
        reps = int(len(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('I', [])[0][0].decode())) / steps)

        for h5_file in h5_files:
            load_data = load_from_h5(os.path.join(data_path, h5_file), 'starkSpec', save_r=1)
            dates.append(datetime.datetime.fromtimestamp(load_data['starkSpec'][self.QubitIndex].get('Dates', [])[0][0]))

            I_shots.append(np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('I', [])[0][0].decode())).reshape(
                    [steps, reps]))
            Q_shots.append(np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Q', [])[0][0].decode())).reshape(
                    [steps, reps]))
            P.append(np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('P', [])[0][0].decode())))

        return dates, n, gain_sweep, steps, reps, I_shots, Q_shots, P

    def plot_shots(self, I_shots, Q_shots, gains, n, round=0, idx=10):
        print(np.shape(I_shots))

        this_I = I_shots[round][idx,:]
        this_Q = Q_shots[round][idx,:]

        print(np.shape(this_I))

        i_new = this_I * np.cos(self.theta) - this_Q * np.sin(self.theta)
        q_new = this_I * np.sin(self.theta) + this_Q * np.cos(self.theta)

        states = (i_new > self.threshold)

        fig, ax = plt.subplots()
        ax.scatter(i_new, q_new, c=states)
        ax.set_xlabel('I [a.u.]')
        ax.set_ylabel('Q [a.u.]')
        ax.set_title(f'dataset {self.dataset} qubit {self.QubitIndex} round {round + 1} of {n}: rotated I,Q shots for stark_spec at gain: {np.round(gains[idx],2)} us')
        plt.show()

    def process_shots(self, I_shots, Q_shots, n, steps):

        p_excited = []
        for round in np.arange(n):
            p_excited_in_round = []
            for idx in np.arange(steps):
                this_I = I_shots[round][idx,:]
                this_Q = Q_shots[round][idx,:]

                i_new = this_I * np.cos(self.theta) - this_Q * np.sin(self.theta)
                q_new = this_I * np.sin(self.theta) + this_Q * np.cos(self.theta)
                if self.thresholding:
                    states = (i_new > self.threshold)
                else:
                    states = np.mean(i_new)
                p_excited_in_round.append(np.mean(states))

            p_excited.append(p_excited_in_round)

        return p_excited

    def gain2freq(self, gains):
        steps = int(len(gains)/2)
        gains_pos_detuning = gains[steps:]
        gains_neg_detuning = gains[:steps]

        # positive detuning, negative frequency shift
        freq_pos_detuning = self.duffing_constant * (self.anharmonicity * np.square(gains_pos_detuning)) / (-1*self.detuning * (self.anharmonicity - self.detuning))

        # negative detuning, positive frequency shift
        freq_neg_detuning = self.duffing_constant * (self.anharmonicity * np.square(gains_neg_detuning)) / (self.detuning * (self.anharmonicity + self.detuning))

        freqs = np.concatenate((freq_neg_detuning, freq_pos_detuning))
        return freqs

    def get_p_excited_in_round(self, gains, p_excited, n, round, plot=True):
        p_excited_in_round = p_excited[round][:]

        if plot:
            fig, ax = plt.subplots(2,1, layout='constrained')
            fig.suptitle(f'dataset {self.dataset} qubit {self.QubitIndex + 1} round {round + 1} of {n} resonator stark spectroscopy')

            ax[0].plot(gains, p_excited_in_round)
            ax[0].set_xlabel('resonator stark gain [a.u.]')
            ax[0].set_ylabel('P(e)')

            ax[1].plot(self.gain2freq(gains), p_excited_in_round)
            ax[1].set_xlabel('stark shift [MHz]')
            ax[1].set_ylabel('P(e)')
            plt.show()

        return p_excited_in_round

class auto_threshold:
    def __init__(self, data_dir, dataset, QubitIndex, folder = "study_data", expt_name = "starkspec_ge"):
        self.data_dir = data_dir
        self.dataset = dataset
        self.QubitIndex = QubitIndex
        self.expt_name = expt_name
        self.folder = folder

    def load_sample(self, idx=0):
        data_path = os.path.join(self.data_dir, self.dataset, self.folder, "Data_h5", self.expt_name)
        h5_files = os.listdir(data_path)
        h5_files.sort()

        load_data = load_from_h5(os.path.join(data_path, h5_files[idx]), 'starkSpec', save_r=1)
        gain_sweep = process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Gain Sweep', [])[0][0].decode())
        steps = len(gain_sweep)
        step_idx = int(steps/2)
        reps = int(len(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('I', [])[0][0].decode())) / steps)
        I_shots = np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('I', [])[0][0].decode())).reshape([steps, reps])
        Q_shots = np.array(process_h5_data(load_data['starkSpec'][self.QubitIndex].get('Q', [])[0][0].decode())).reshape([steps, reps])

        return I_shots[step_idx], Q_shots[step_idx]

    def get_threshold(self, I_shots, Q_shots, plot=True):
        kmeans = KMeans(n_clusters=2).fit(np.transpose([I_shots, Q_shots]))
        print(kmeans.cluster_centers_[:,0])
        ye = kmeans.cluster_centers_[1,1]
        xe = kmeans.cluster_centers_[1,0]
        yg = kmeans.cluster_centers_[0,1]
        xg = kmeans.cluster_centers_[0,0]

        theta = -np.arctan2((ye - yg), (xe - xg))
        i_new = I_shots * np.cos(theta) - Q_shots * np.sin(theta)
        q_new = I_shots * np.sin(theta) + Q_shots * np.cos(theta)

        kmeans_new = KMeans(n_clusters=2).fit(np.transpose([i_new, q_new]))
        threshold = np.mean([kmeans_new.cluster_centers_[1,0],kmeans_new.cluster_centers_[0,0]])
        state = kmeans_new.predict(np.transpose([i_new, q_new]))

        if plot:
            fig, ax = plt.subplots(1,2, layout='constrained')

            plot = ax[0]
            plot.scatter(I_shots, Q_shots, c=state)
            plot.set_xlabel('I [a.u.]')
            plot.set_ylabel('Q [a.u.]')
            plot.scatter(xg, yg, c='k')
            plot.scatter(xe, ye, c='k')
            plot.set_aspect('equal')
            plot.set_title('unrotated I,Q')

            plot = ax[1]
            plot.scatter(i_new, q_new, c=state)
            plot.set_xlabel('I [a.u.]')
            plot.set_ylabel('Q [a.u.]')
            plot.set_aspect('equal')
            plot.set_title(f'rotated I,Q; theta={np.round(theta,2)}, threshold={np.round(threshold,2)}')
            plot.plot([threshold, threshold], [np.min(q_new), np.max(q_new)], 'k:')

            plt.show()


        return theta, threshold, i_new, q_new
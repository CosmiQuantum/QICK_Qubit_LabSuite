from qicklab import *

save_plot_top_folder = '/Users/olivias-local/Downloads/source_off_plots/'
create_folder_if_not_exists(save_plot_top_folder)
base_path = "/Users/olivias-local/Downloads/source_off/"

number_of_qubits = 6
#----------------------------------------------- SSF plots --------------------------------------------------------
subfolder_path = "optimization/Data_h5/SS_ge/"
filepaths_ss_spec = []
for folder in os.listdir(base_path):
    full_path = os.path.join(base_path, folder, subfolder_path)
    if os.path.isdir(full_path):
        filepaths_ss_spec.append(full_path)
filepaths_ss_spec = get_h5_files_in_dirs(filepaths_ss_spec)

fidelities, angles, thresholds, date_times = get_ss_info_and_dates(number_of_qubits, filepaths_ss_spec, outer_folder=save_plot_top_folder + 'SS', save_figs=False)
scatter_plot_vs_time_with_fit_errs(date_times, angles, number_of_qubits, y_data_name = 'Rotation theta',
                                   y_label = 'theta (deg)', save_name = 'ssf_angles',
                                   save_folder_path = save_plot_top_folder + 'SS',
                                   show_legends = False, final_figure_quality = 100)
gaussian_data, means, stds = plot_histogram_with_gaussian(data_dict=angles, date_dict=date_times,
                                                          save_folder_path=save_plot_top_folder + 'SS',
                                                          save_name='ssf_angles',
                                                          data_type='Angles',
                                                          x_label='theta (deg)', y_label='Frequency')
scatter_plot_vs_time_with_fit_errs(date_times, thresholds, number_of_qubits, y_data_name = 'I threshold',
                                   y_label = 'Threshold (a.u)', save_name = 'ssf_thresholds',
                                   save_folder_path = save_plot_top_folder + 'SS',
                                   show_legends = False, final_figure_quality = 100)
gaussian_data, means, stds = plot_histogram_with_gaussian(data_dict=angles, date_dict=date_times,
                                                          save_folder_path=save_plot_top_folder + 'SS',
                                                          save_name='ssf_thresholds',
                                                          data_type='Thresholds',
                                                          x_label='Threshold (a.u)', y_label='Frequency')

#----------------------------------------------- Res spec plots --------------------------------------------------------
subfolder_path = "optimization/Data_h5/Res_ge/"
filepaths_r_spec = []
for folder in os.listdir(base_path):
    full_path = os.path.join(base_path, folder, subfolder_path)
    if os.path.isdir(full_path):
        filepaths_r_spec.append(full_path)
filepaths_r_spec = get_h5_files_in_dirs(filepaths_r_spec)


frequencies, date_times = get_res_freqs_and_dates(number_of_qubits, filepaths_r_spec, outer_folder=save_plot_top_folder + 'RSpec', save_figs=False)

scatter_plot_vs_time_with_fit_errs(date_times, frequencies, number_of_qubits, y_data_name = 'Res Frequency',
                                   y_label = 'Frequency (MHz)', save_name = 'rspec',
                                   save_folder_path = save_plot_top_folder + 'RSpec',
                                   show_legends = False, final_figure_quality = 100)
gaussian_data, means, stds = plot_histogram_with_gaussian(data_dict=frequencies, date_dict=date_times,
                                                          save_folder_path=save_plot_top_folder + 'RSpec',
                                                          save_name='rfreq',
                                                          data_type='RFreq',
                                                          x_label='Freq (MHz)', y_label='Frequency')
plot_cumulative_distribution(data_dict=frequencies, gaussian_fit_data=gaussian_data,
                             save_folder_path=save_plot_top_folder + 'RSpec',
                             save_name='rfreq',
                             data_type='Res Frequency', x_label='Freq (MHz)', y_label='Cumulative Distribution')

plot_lomb_scargle_spectral_density(date_times, frequencies, number_of_qubits, show_legends=False, label="Res Freq",
                                   save_label='r_freq',
                                   save_folder_path=save_plot_top_folder + 'RSpec',
                                   final_figure_quality=100, log_freqs=False)

plot_allan_deviation_largest_continuous_sample(date_times, frequencies, number_of_qubits, show_legends=True,
                                               label="Res Freq", save_label='r_freq',
                                               save_folder_path=save_plot_top_folder + 'RSpec',
                                               final_figure_quality=100, plot_all_data_segments=True,
                                               stack_segments_xaxis=False, stack_segments_yaxis=False)
plot_welch_spectral_density_largest_continuous_sample(date_times, frequencies, number_of_qubits, show_legends=True,
                                                      save_label='r_freq', label="Res Freq",
                                                      save_folder_path=save_plot_top_folder + 'RSpec',
                                                      final_figure_quality=100, plot_all_data_segments=True,
                                                      stack_segments_xaxis=False, stack_segments_yaxis=False,
                                                      resample=False)

##Look at time differences between qspec samples for the data that is used in the WSD and Allan Dev functions above
dt_diffs_cont_dict = {}
for key in date_times.keys():
    sorted_times, sorted_vals = sort_date_time_data(date_times[key], frequencies[key])
    if len(sorted_times) >0:
        time_sec = convert_datetimes_to_seconds(sorted_times)
        vals_array = np.array(sorted_vals, dtype=float)
        segments_time, segments_vals = split_into_continuous_segments(time_sec, vals_array, gap_threshold_factor=10)
        times, vals = get_longest_continuous_segment(segments_time, segments_vals)
        dt_diffs_cont = np.diff(times)
        dt_diffs_cont_dict[key] = dt_diffs_cont
    else:
        dt_diffs_cont_dict[key] = []

gaussian_data, means, stds = plot_histogram_with_gaussian(data_dict=dt_diffs_cont_dict, date_dict=date_times,
                                                          save_name='rspec_freq_time_diffs',
                                                          save_folder_path = save_plot_top_folder + 'RSpec',
                                                          data_type='Res Freq',
                                                          title='Res Frequency measurement time differences',
                                                          x_label='Time diff (s)',
                                                          y_label='Frequency', bin_count = 10)

del frequencies, date_times

#------------------------------------------------- QFreq plots --------------------------------------------------------
subfolder_path = "optimization/Data_h5/QSpec_ge/"
filepaths_qspec = []
for folder in os.listdir(base_path):
    full_path = os.path.join(base_path, folder, subfolder_path)
    if os.path.isdir(full_path):
        filepaths_qspec.append(full_path)
filepaths_qspec = get_h5_files_in_dirs(filepaths_qspec)


filepaths_qspec = get_h5_files_in_dirs(filepaths_qspec)
frequencies, fit_errs, date_times = get_freqs_and_dates(number_of_qubits, filepaths_qspec,
                                                        outer_folder=save_plot_top_folder + 'QSpec',
                                                        verbose=False, save_figs=False, expt_name='QSpec_ge',
                                                        fit_err_threshold=0.15)

scatter_plot_vs_time_with_fit_errs(date_times, frequencies, number_of_qubits, y_data_name = 'Qubit Frequency',
                                   y_label = 'Frequency (MHz)', save_name = 'qspec',
                                   save_folder_path = save_plot_top_folder  + 'QSpec',
                                   show_legends = False, fit_err=fit_errs, final_figure_quality = 100)
gaussian_data, means, stds = plot_histogram_with_gaussian(data_dict=frequencies, date_dict=date_times,
                                                          save_folder_path=save_plot_top_folder + 'QSpec',
                                                          save_name='qfreq',
                                                          data_type='QFreq',
                                                          x_label='Freq (MHz)', y_label='Frequency')
plot_cumulative_distribution(data_dict=frequencies, gaussian_fit_data=gaussian_data,
                             save_folder_path=save_plot_top_folder + 'QSpec',
                             save_name='qfreq',
                             data_type='Qubit Frequency', x_label='Freq (MHz)', y_label='Cumulative Distribution')
plot_error_vs_value(data_dict=frequencies, error_dict=fit_errs,
                    save_folder_path=save_plot_top_folder+ 'QSpec',
                    save_name='qfreq',
                    data_type='Qubit Frequency',
                    x_label='Frequency (MHz)', y_label='Fit error (MHz)')

plot_lomb_scargle_spectral_density(date_times, frequencies, number_of_qubits, show_legends=False, label="Qubit Freq",
                                   save_label='q_freq',
                                   save_folder_path=save_plot_top_folder+ 'QSpec',
                                   final_figure_quality=100, log_freqs=False)

plot_allan_deviation_largest_continuous_sample(date_times, frequencies, number_of_qubits, show_legends=True,
                                               label="Qubit Freq", save_label='q_freq',
                                               save_folder_path=save_plot_top_folder+ 'QSpec',
                                               final_figure_quality=100, plot_all_data_segments=True,
                                               stack_segments_xaxis=False, stack_segments_yaxis=False)
plot_welch_spectral_density_largest_continuous_sample(date_times, frequencies, number_of_qubits, show_legends=True,
                                                      save_label='q_freq', label="Qubit Freq",
                                                      save_folder_path=save_plot_top_folder+ 'QSpec',
                                                      final_figure_quality=100, plot_all_data_segments=True,
                                                      stack_segments_xaxis=False, stack_segments_yaxis=False,
                                                      resample=False)

##Look at time differences between qspec samples for the data that is used in the WSD and Allan Dev functions above
dt_diffs_cont_dict = {}
for key in date_times.keys():
    sorted_times, sorted_vals = sort_date_time_data(date_times[key], frequencies[key])
    if len(sorted_times) >0:
        time_sec = convert_datetimes_to_seconds(sorted_times)
        vals_array = np.array(sorted_vals, dtype=float)
        segments_time, segments_vals = split_into_continuous_segments(time_sec, vals_array, gap_threshold_factor=10)
        times, vals = get_longest_continuous_segment(segments_time, segments_vals)
        dt_diffs_cont = np.diff(times)
        dt_diffs_cont_dict[key] = dt_diffs_cont
    else:
        dt_diffs_cont_dict[key] = []

gaussian_data, means, stds = plot_histogram_with_gaussian(data_dict=dt_diffs_cont_dict, date_dict=date_times,
                                                          save_name='qspec_freq_time_diffs',
                                                          save_folder_path = save_plot_top_folder+ 'QSpec',
                                                          title='Qubit Frequency measurement time differences',
                                                          x_label='Time diff (s)',
                                                          y_label='Frequency', bin_count = 10)

del frequencies, fit_errs, date_times

##-------------------------------------------------- T1 plots ----------------------------------------------------------
subfolder_path = "study/Data_h5/T1_ge/"
filepaths_T1 = []
for folder in os.listdir(base_path):
    full_path = os.path.join(base_path, folder, subfolder_path)
    if os.path.isdir(full_path):
        filepaths_T1.append(full_path)
filepaths_T1 = get_h5_files_in_dirs(filepaths_T1)


decoherence_times, fit_errs, date_times, decoherence_times_good_data, fit_errs_good_data, date_times_good_data\
    = get_decoherence_time_and_dates(number_of_qubits, filepaths_T1,
                                                                         outer_folder=save_plot_top_folder+ 'T1',
                                                                         decoherence_type = 'T1',
                                                                         expt_name='T1_ge',
                                                                         discard_values_over = 300,
                                                                         fit_err_threshold=40, save_figs=False,
                                                                         thresholding=True,
                                                                         discard_low_signal_values = True)

scatter_plot_vs_time_with_fit_errs(date_times, decoherence_times,  number_of_qubits, y_data_name = 'T1 Decay',
                                   y_label = 'Relaxation time (us)', save_name = 't1',
                                   save_folder_path = save_plot_top_folder+ 'T1',
                                   show_legends = False,fit_err=fit_errs, final_figure_quality = 100)
gaussian_data, means, stds = plot_histogram_with_gaussian(data_dict=decoherence_times, date_dict=date_times,
                                                          save_name='t1',
                                                          save_folder_path = save_plot_top_folder+ 'T1',
                                                          data_type='T1', x_label='T1 (µs)',
                                                          y_label='Frequency')
plot_cumulative_distribution(decoherence_times, gaussian_fit_data=gaussian_data, save_name='t1',
                             save_folder_path=save_plot_top_folder+ 'T1',
                             data_type='T1', x_label='T1 (µs)', y_label='Cumulative Distribution')
plot_error_vs_value(decoherence_times, error_dict=fit_errs, save_name = 't1',
                    save_folder_path = save_plot_top_folder+ 'T1',
                    x_label='T1 (µs)', y_label='Fit error (µs)')
plot_lomb_scargle_spectral_density(date_times_good_data, decoherence_times_good_data, number_of_qubits, show_legends=False,
                                save_label='t1', label="T1",
                                save_folder_path=save_plot_top_folder+ 'T1',
                                final_figure_quality=100,
                                log_freqs=False)

plot_allan_deviation_largest_continuous_sample(date_times_good_data, decoherence_times_good_data, number_of_qubits, show_legends=True,
                                               save_label='t1', label="T1",
                                               save_folder_path=save_plot_top_folder+ 'T1',
                                               final_figure_quality=100, plot_all_data_segments=True,
                                               stack_segments_xaxis=False, stack_segments_yaxis=False, fit=False)
plot_welch_spectral_density_largest_continuous_sample(date_times_good_data, decoherence_times_good_data, number_of_qubits,
                                                      show_legends=True, save_label='t1', label="T1",
                                                      save_folder_path=save_plot_top_folder+ 'T1',
                                                      final_figure_quality=100, plot_all_data_segments=True,
                                                      stack_segments_xaxis=False, stack_segments_yaxis=False,
                                                      resample=False)

##Look at time differences between t1 samples for the data that is used in the WSD and Allan Dev functions above
dt_diffs_cont_dict = {}
for key in date_times.keys():
    sorted_times, sorted_vals = sort_date_time_data(date_times_good_data[key], decoherence_times_good_data[key])
    if len(sorted_times) > 0:
        time_sec = convert_datetimes_to_seconds(sorted_times)
        vals_array = np.array(sorted_vals, dtype=float)
        segments_time, segments_vals = split_into_continuous_segments(time_sec, vals_array, gap_threshold_factor=10)
        times, vals = get_longest_continuous_segment(segments_time, segments_vals)
        dt_diffs_cont = np.diff(times)
        dt_diffs_cont_dict[key] = dt_diffs_cont
    else:
        dt_diffs_cont_dict[key] = []

gaussian_data, means, stds = plot_histogram_with_gaussian(data_dict=dt_diffs_cont_dict, date_dict=date_times,
                                                          save_name='t1_measurement_time_diffs',
                                                          save_folder_path = save_plot_top_folder+ 'T1',
                                                          data_type='T1', title='T1 measurement time differences',
                                                          x_label='Time diff (s)',
                                                          y_label='Frequency', bin_count = 10)
del decoherence_times, fit_errs, date_times

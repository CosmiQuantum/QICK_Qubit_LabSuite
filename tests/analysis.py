from qicklab import *

filepaths = get_h5_files_in_dirs( ["/Users/olivias-local/Downloads/TLS_Studies/Q0/Data_h5/T1_ge/",
                                   "/Users/olivias-local/Downloads/TLS_Studies/Q1/Data_h5/T1_ge/",
                                   "/Users/olivias-local/Downloads/TLS_Studies/Q2/Data_h5/T1_ge/",
                                   "/Users/olivias-local/Downloads/TLS_Studies/Q3/Data_h5/T1_ge/",
                                   "/Users/olivias-local/Downloads/TLS_Studies/Q4/Data_h5/T1_ge/",
                                  "/Users/olivias-local/Downloads/TLS_Studies/Q5/Data_h5/T1_ge/"])
number_of_qubits = 6
##------------------------------------------------- QFreq plots --------------------------------------------------------
# frequencies, fit_errs, date_times = get_freqs_and_dates(number_of_qubits, filepaths)
# scatter_plot_vs_time_with_fit_errs(date_times, frequencies, fit_errs, number_of_qubits, y_data_name = 'Qubit Frequency',
#                                    y_label = 'Frequency (MHz)', save_name = 'qspec',
#                                    save_folder_path = '/Users/olivias-local/Downloads/Test_plotting/',
#                                    show_legends = False, final_figure_quality = 100)
# gaussian_data, means, stds = plot_histogram_with_gaussian(data_dict=frequencies, date_dict=date_times,
#                                                           save_folder_path='/Users/olivias-local/Downloads/Test_plotting/',
#                                                           save_name='qfreq',
#                                                           data_type='QFreq',
#                                                           x_label='Freq (MHz)', y_label='Frequency')
# plot_cumulative_distribution(data_dict=frequencies, gaussian_fit_data=gaussian_data,
#                              save_folder_path='/Users/olivias-local/Downloads/Test_plotting/',
#                              save_name='qfreq',
#                              data_type='Qubit Frequency', x_label='Freq (MHz)', y_label='Cumulative Distribution')
# plot_error_vs_value(data_dict=frequencies, error_dict=fit_errs,
#                     save_folder_path='/Users/olivias-local/Downloads/Test_plotting/',
#                     save_name='qfreq',
#                     data_type='Qubit Frequency',
#                     x_label='Frequency (MHz)', y_label='Fit error (MHz)')
# del frequencies, fit_errs, date_times

##-------------------------------------------------- T1 plots ----------------------------------------------------------
decoherence_times, fit_errs, date_times = get_decoherence_time_and_dates(number_of_qubits, filepaths,
                                                                         decoherence_type = 'T1',
                                                                         discard_values_over = 300)

# scatter_plot_vs_time_with_fit_errs(date_times, decoherence_times, fit_errs, number_of_qubits, y_data_name = 'T1 Decay',
#                                    y_label = 'Relaxation time (us)', save_name = 't1',
#                                    save_folder_path = '/Users/olivias-local/Downloads/Test_plotting/',
#                                    show_legends = False, final_figure_quality = 100)
# gaussian_data, means, stds = plot_histogram_with_gaussian(data_dict=decoherence_times, date_dict=date_times,
#                                                           save_name='t1',
#                                                           save_folder_path = '/Users/olivias-local/Downloads/Test_plotting/',
#                                                           data_type='T1', x_label='T1 (µs)',
#                                                           y_label='Frequency')
# plot_cumulative_distribution(decoherence_times, gaussian_fit_data=gaussian_data, save_name='t1',
#                              data_type='T1', x_label='T1 (µs)', y_label='Cumulative Distribution')
# plot_error_vs_value(decoherence_times, error_dict=fit_errs, save_name = 't1',
#                     save_folder_path = '/Users/olivias-local/Downloads/Test_plotting/',
#                     x_label='T1 (µs)', y_label='Fit error (µs)')
# plot_welch_spectral_density(date_times, decoherence_times, number_of_qubits, show_legends=False, label="T1",
#                                 save_folder_path='/Users/olivias-local/Downloads/Test_plotting/', final_figure_quality=100) #assumes uniform
# plot_allan_deviation(date_times, decoherence_times, number_of_qubits, show_legends=False, label="T1",
#                          save_folder_path='/Users/olivias-local/Downloads/Test_plotting/', final_figure_quality=100) #assumes uniform
# plot_lomb_scargle_spectral_density(date_times, decoherence_times, number_of_qubits, show_legends=False, label="T1",
#                                 save_folder_path='/Users/olivias-local/Downloads/Test_plotting/', final_figure_quality=100,
#                                 log_freqs=False)
#
# plot_allan_deviation_largest_continuous_sample(date_times, decoherence_times, number_of_qubits, show_legends=False, label="T1",
#                          save_folder_path='/Users/olivias-local/Downloads/Test_plotting/', final_figure_quality=100)
# plot_welch_spectral_density_largest_continuous_sample(date_times, decoherence_times, number_of_qubits, show_legends=False, label="T1",
#                                                       save_folder_path='/Users/olivias-local/Downloads/Test_plotting/', final_figure_quality=100)

# Look at time differences between t1 samples for the data that is used in the WSD and Allan Dev functions above
dt_diffs_cont_dict = {}
for key in date_times.keys():
    sorted_times, sorted_vals = sort_date_time_data(date_times[key], decoherence_times[key])
    time_sec = convert_datetimes_to_seconds(sorted_times)
    vals_array = np.array(sorted_vals, dtype=float)
    segments_time, segments_vals = split_into_continuous_segments(time_sec, vals_array, gap_threshold_factor=5)
    times, vals = get_longest_continuous_segment(segments_time, segments_vals)
    dt_diffs_cont = np.diff(times)
    dt_diffs_cont_dict[key] = dt_diffs_cont

gaussian_data, means, stds = plot_histogram_with_gaussian(data_dict=dt_diffs_cont_dict, date_dict=date_times,
                                                          save_name='t1_measurement_time_diffs',
                                                          save_folder_path = '/Users/olivias-local/Downloads/Test_plotting/',
                                                          data_type='T1', x_label='T1 measurement time differences (s)',
                                                          y_label='Frequency', bin_count = 10)


del decoherence_times, fit_errs, date_times



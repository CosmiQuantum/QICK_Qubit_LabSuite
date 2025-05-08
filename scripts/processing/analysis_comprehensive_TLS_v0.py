import numpy as np
import matplotlib.pyplot as plt

from qicklab.analysis import AnaQSpec, t1, ssf, AnaResStarkSpec, starkspec, AnaAutoThreshold
from qicklab.utils import get_abs_min

############### set values here ###################
data_dir = "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy1/" #update based on file transfer location from Ryan
dataset = '2025-04-15_21-24-46'

QubitIndex = 0 #zero indexed
analysis_flags = {"get_threshold": True, "load_all_data": True, "timestream": True, "round": True}
selected_round = [10, 73]
threshold = 0 #overwritten when get_threshold flag is set to True
theta = 0 #overwritten when get_threshold flag is set to True

res_stark_constant = [-17, 0, 0, 0, -25, 0] # from res stark 2D map
duffing_constant = [220, 1, 1, 1, 100, 1] # from stark 2D map at fixed detunings
anharmonicity = [-173.65, -177.3, -173.74, -171.45, -155.9, -165.9] #from run6 hamiltonian spreadsheet
detuning = [20, 10, 10, 10, 10, 10]

if analysis_flags["get_threshold"]:
    print("Determining threshold...")

    ana_params = {
        "idx" : 0,
        "plot": True,
    }

    auto = AnaAutoThreshold(data_dir, dataset, QubitIndex, ana_params=ana_params)

    data = auto.load_all()
    result = auto.run_analysis(verbose=True)

    I_shots = data["I"]
    Q_shots = data["Q"]

    theta = result["theta"]
    threshold = result["threshold"]
    i_new = result["I_new"]
    q_new = result["Q_new"]

    auto.cleanup()
    del auto

if analysis_flags["load_all_data"]:
    
    ## =============================== QSPEC =============================== ##
    print("Loading qspec data...")

    ana_params = {
        "plot_idxs": selected_round,
    }

    qspec_ge = AnaQSpec(data_dir, dataset, QubitIndex, ana_params=ana_params)
    
    data = qspec_ge.load_all()
    result = qspec_ge.run_analysis(verbose=True)

    start_time = data["dates"][0]
    qspec_dates = data["dates"]
    qspec_probe_freqs = data["qspec_probe_freqs"]
    qspec_I = data["I"]
    qspec_Q = data["Q"]

    qspec_freqs = result["qspec_freqs"]
    qspec_errs = result["qspec_errs"]
    qspec_fwhms = result["qspec_fwhms"]

    qspec_ge.cleanup()
    del qspec_ge

    ## =============================== SSF =============================== ##
    print("Loading SSF data...")
    ssf_ge = ssf(data_dir, dataset, QubitIndex)
    ssf_dates, ssf_n, I_g, Q_g, I_e, Q_e, fid, angles = ssf_ge.load_all()

    print("Loading T1 data...")
    t1_ge = t1(data_dir, dataset, QubitIndex, theta, threshold)
    t1_dates, t1_n, delay_times, t1_steps, t1_reps, t1_I_shots, t1_Q_shots = t1_ge.load_all()
    t1_p_excited = t1_ge.process_shots(t1_I_shots, t1_Q_shots, t1_n, t1_steps)
    t1s, t1_errs = t1_ge.get_all_t1(delay_times, t1_p_excited, t1_n)

    print("Loading HGqspec data...")
    hgqspec = qspec(data_dir, dataset, QubitIndex, expt_name="high_gain_qspec_ge")
    hgqspec_dates, hgqspec_n, hgqspec_probe_freqs, hgqspec_I, hgqspec_Q = hgqspec.load_all()
    hgqspec_mag = np.sqrt(np.square(hgqspec_I) + np.square(hgqspec_Q))

    ## =============================== ResStark =============================== ##
    print("Loading resonator Stark spec data...")
    ana_params = {
        "theta": theta,
        "threshold": threshold,
        "stark_constant": res_stark_constant[QubitIndex],
        "thresholding": False
    }

    rstark = AnaResStarkSpec(data_dir, dataset, QubitIndex, ana_params=ana_params)

    data = rstark.load_all()
    result = rstark.run_analysis(verbose=True)

    rstark_dates = data["dates"]
    rstark_n = data["n"]
    rstark_gains = data["gain_sweep"]
    rstark_p_excited = result["rstark_p_excited"]
    rstark_freqs = result["rstark_freqs"]

    rstark.cleanup()
    del rstark
    

    print("Loading detuning Stark spec data...")
    stark = starkspec(data_dir, dataset, QubitIndex, duffing_constant[QubitIndex], theta, threshold, anharmonicity[QubitIndex], detuning[QubitIndex])
    stark_dates, stark_n, stark_gains, stark_steps, stark_reps, stark_I_shots, stark_Q_shots, stark_P = stark.load_all()
    stark_p_excited = stark.process_shots(stark_I_shots, stark_Q_shots, stark_n, stark_steps)
    stark_freqs = stark.gain2freq(stark_gains)


if analysis_flags["timestream"]:
    print("Generating timestream plots...")

    fig, ax = plt.subplots(3,2, layout='constrained')
    fig.suptitle(f'dataset {dataset} qubit {QubitIndex + 1} timestream')

    #### low gain qspec ####
    plot = ax[0][0]
    plot.errorbar(get_abs_min(start_time,qspec_dates), qspec_freqs, qspec_errs, fmt='o')
    plot.set_xlabel('time [min]')
    plot.set_ylabel('qspec frequency [MHz]')
    for i in selected_round:
        plot.scatter((qspec_dates[i] - start_time).total_seconds()/60, qspec_freqs[i], marker="o",s=200, alpha=0.5)

    ##### single-shot ##########
    plot = ax[1][0]
    plot.errorbar(get_abs_min(start_time,ssf_dates), np.array(fid) * 100, fmt='o')
    plot.set_xlabel('time [min]')
    plot.set_ylabel('single-shot fidelity [%]')
    for i in selected_round:
        plot.scatter((ssf_dates[i] - start_time).total_seconds()/60, fid[i]*100, marker="o",s=200, alpha=0.5)

    ##### t1 data #####
    plot = ax[2][0]
    plot.errorbar(get_abs_min(start_time,t1_dates), t1s, t1_errs, fmt='o')
    plot.set_xlabel('time [min]')
    plot.set_ylabel('t1_ge [us]')
    for i in selected_round:
        plot.scatter((t1_dates[i] - start_time).total_seconds()/60, t1s[i], marker="o",s=200, alpha=0.5)

    ##### high gain qspec #####
    plot = ax[0][1]
    cbar = plt.colorbar(plot.pcolormesh(get_abs_min(start_time,hgqspec_dates), hgqspec_probe_freqs , np.transpose(hgqspec_mag), #np.transpose(np.sqrt(np.square(I)+np.square(Q))),
                                        shading="nearest", cmap="viridis"), ax=plot)
    cbar.set_label("I,Q magnitude [a.u.]")
    plot.set_ylabel('qubit probe frequency [MHz]')
    plot.set_xlabel('time [min]')
    plot.set_title('high gain qspec_ge')
    for i in selected_round:
        plot.scatter((hgqspec_dates[i] - start_time).total_seconds()/60, hgqspec_probe_freqs[0], marker="^",s=200, alpha=1.0)

    ##### res stark spec ######
    plot = ax[1][1]
    cbar = plt.colorbar(plot.pcolormesh(get_abs_min(start_time,rstark_dates), rstark_freqs, np.transpose(rstark_p_excited),
                                        shading="nearest", cmap="viridis"), ax=plot)
    cbar.set_label("P(MS=1)")
    plot.set_ylabel('stark shift [MHz]')
    plot.set_xlabel('time [min]')
    plot.set_title('stark tone @ resonator frequency')
    for i in selected_round:
        plot.scatter((rstark_dates[i] - start_time).total_seconds()/60, rstark_freqs[len(rstark_freqs)-1], marker="^",s=150, alpha=1.0)

    #### stark spec at fixed detuning #####
    plot = ax[2][1]
    cbar = plt.colorbar(plot.pcolormesh(get_abs_min(start_time,stark_dates), stark_freqs, np.transpose(stark_p_excited),
                                        shading="nearest", cmap="viridis"), ax=plot)
    cbar.set_label("P(MS=1)")
    plot.set_ylabel('stark shift [MHz]')
    plot.set_xlabel('time [min]')
    plot.set_title('stark tone @ fixed detuning from qubit frequency')
    for i in selected_round:
        plot.scatter((stark_dates[i] - start_time).total_seconds()/60, stark_freqs[len(stark_freqs)-1], marker="^",s=150, alpha=1.0)

    #plt.show(block=False)


if analysis_flags['round']:
    print("Generating round-robin plots...")
    fig, ax = plt.subplots(2, 3, layout='constrained')
    plt.rcParams['lines.linewidth'] = 1

    for round in selected_round:
        ##### qspec data ####
        qspec_mag = np.sqrt(np.square(qspec_I)+np.square(qspec_Q))
        qfreq, qfreq_err, fwhm, qspec_fit = qspec_ge.get_qspec_freq_in_round(qspec_probe_freqs, qspec_I, qspec_Q, round, qspec_n, plot=False)

        plot = ax[0][0]
        plot.plot(qspec_probe_freqs, qspec_mag[round], label=f'round {round + 1}')
        plot.plot(qspec_probe_freqs, qspec_fit, 'k:')
        plot.set_xlabel('qubit probe frequency [MHz]')
        plot.set_ylabel('I,Q magnitude [a.u.]')
        plot.set_title('low gain qspec_ge')
        plot.legend()

        ##### high gain qspec #####

        plot = ax[0][1]
        plot.plot(hgqspec_probe_freqs, hgqspec_mag[round], label=f'round {round + 1}')
        plot.legend()
        plot.set_xlabel('qubit probe frequency [MHz]')
        plot.set_ylabel('I,Q magnitude [a.u.]')
        plot.set_title('high gain qspec_ge')

        ##### t1 data #####
        q1_fit_exponential, T1_err, T1_est = t1_ge.get_t1_in_round(delay_times, t1_p_excited, t1_n, round, plot = False)

        plot = ax[0][2]
        plot.plot(delay_times, t1_p_excited[round], label=f'round {round + 1} T1 = {T1_est:.2f} +/- {T1_err:.2f} us')
        plot.plot(delay_times, q1_fit_exponential, 'k:')
        plot.set_title('t1_ge')
        plot.set_ylabel('P(e)')
        plot.set_xlabel('delay time [us]')


        rstark_p_excited_in_round = rstark.get_p_excited_in_round(rstark_gains, rstark_p_excited, rstark_n, round, plot = False)
        plot = ax[1][0]
        plot.plot(rstark_freqs, rstark_p_excited_in_round, label=f'round {round + 1}')
        plot.set_xlabel('stark shift [MHz]')
        plot.set_ylabel('P(e)')
        plot.set_title('stark tone @ resonator frequency')
        plot.legend()

        stark_p_excited_in_round = stark.get_p_excited_in_round(stark_gains, stark_p_excited, stark_n, round, plot = False)

        plot = ax[1][1]
        plot.plot(stark_freqs, stark_p_excited_in_round, label=f'round {round + 1}')
        plot.set_xlabel('stark shift [MHz]')
        plot.set_ylabel('P(e)')
        plot.set_title('stark tone @ fixed detuning')
        plot.legend()


        # ig_new, qg_new, ie_new, qe_new, xg, yg, xe, ye = ssf_ge.get_ssf_in_round(I_g, Q_g, I_e, Q_e, round)
        #
        # plot = ax[1][2]
        # plot.scatter(ig_new, qg_new, s=2, color='b', label='g')
        # plot.scatter(ie_new, qe_new, s=2, color='r', label='e')
        # plot.set_xlabel('I [a.u.]')
        # plot.set_ylabel('Q [a.u.]')
        # plot.legend()


    fig.suptitle(f'dataset {dataset} qubit {QubitIndex + 1}')

plt.show()





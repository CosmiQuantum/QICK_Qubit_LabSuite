import numpy as np
import matplotlib.pyplot as plt

from qicklab.analysis import rspec, qspec, t1, ssf, resstarkspec, starkspec, auto_threshold, ampRabi
from qicklab.utils import get_abs_min

############### set values here ###################
data_dir = "/Users/joycecs/PycharmProjects/PythonProject/.venv/QUIET/QUIET_data/RR_comprehensive_TLS/" #update based on file transfer location from Ryan
#data_dir = "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/source_off_substudy1/" #update based on file transfer location from Ryan
substudy = 'source_off_substudy1'
dataset = '2025-04-15_21-24-46'

QubitIndex = 0 #zero indexed
analysis_flags = {"get_threshold": False, "load_all_data": False, "load_optimization_data": True, "timestream": False, "round": False, "g-e_optimization_report": True}
selected_round = [10, 73]
threshold = 0 #overwritten when get_threshold flag is set to True
theta = 0 #overwritten when get_threshold flag is set to True
sz=10 #fontsize for plots

res_stark_constant = [-17, 0, 0, 0, -25, 0] # from res stark 2D map
duffing_constant = [220, 1, 1, 1, 100, 1] # from stark 2D map at fixed detunings
anharmonicity = [-173.65, -177.3, -173.74, -171.45, -155.9, -165.9] #from run6 hamiltonian spreadsheet
detuning = [20, 10, 10, 10, 10, 10]

if analysis_flags["get_threshold"]:
    print("Determining threshold...")
    auto = auto_threshold(data_dir, dataset, QubitIndex)
    I_shots, Q_shots = auto.load_sample()
    theta, threshold, i_new, q_new = auto.get_threshold(I_shots, Q_shots, plot=True)

if analysis_flags["load_all_data"]:

    try:
        print("Loading qspec data...")
        qspec_ge = qspec(data_dir, dataset, QubitIndex)
        qspec_dates, qspec_n, qspec_probe_freqs, qspec_I, qspec_Q = qspec_ge.load_all()
        qspec_freqs, qspec_errs, qspec_fwhms = qspec_ge.get_all_qspec_freq(qspec_probe_freqs, qspec_I, qspec_Q, qspec_n)
        start_time = qspec_dates[0]
    except Exception:
        print("qspec_ge data error or qspec_ge data missing in data-taking block")


    try:
        print("Loading SSF data...")
        ssf_ge = ssf(data_dir, dataset, QubitIndex)
        ssf_dates, ssf_n, I_g, Q_g, I_e, Q_e, fid, angles = ssf_ge.load_all()
    except Exception:
        print("ssf_ge data error or ssf_ge data missing in data-taking block")

    try:
        print("Loading T1 data...")
        t1_ge = t1(data_dir, dataset, QubitIndex, theta, threshold)
        t1_dates, t1_n, delay_times, t1_steps, t1_reps, t1_I_shots, t1_Q_shots = t1_ge.load_all()
        t1_p_excited = t1_ge.process_shots(t1_I_shots, t1_Q_shots, t1_n, t1_steps)
        t1s, t1_errs = t1_ge.get_all_t1(delay_times, t1_p_excited, t1_n)
    except Exception:
        print("t1_ge data error or t1_ge data missing in data-taking block")

    try:
        print("Loading HGqspec data...")
        hgqspec = qspec(data_dir, dataset, QubitIndex, expt_name="high_gain_qspec_ge")
        hgqspec_dates, hgqspec_n, hgqspec_probe_freqs, hgqspec_I, hgqspec_Q = hgqspec.load_all()
        hgqspec_mag = np.sqrt(np.square(hgqspec_I) + np.square(hgqspec_Q))
    except Exception:
        print("high gain qspec_ge data error or high gain qspec_ge data missing in data-taking block")

    try:
        print("Loading resonator Stark spec data...")
        rstark = resstarkspec(data_dir, dataset, QubitIndex, res_stark_constant[QubitIndex], theta, threshold)
        rstark_dates, rstark_n, rstark_gains, rstark_steps, rstark_reps, rstark_I_shots, rstark_Q_shots, rstark_P = rstark.load_all()
        rstark_p_excited = rstark.process_shots(rstark_I_shots, rstark_Q_shots, rstark_n, rstark_steps)
        rstark_freqs = rstark.gain2freq(rstark_gains)
    except Exception:
        print("res stark spec data error or res stark spec data missing in data-taking block")

    try:
        print("Loading detuning Stark spec data...")
        stark = starkspec(data_dir, dataset, QubitIndex, duffing_constant[QubitIndex], theta, threshold, anharmonicity[QubitIndex], detuning[QubitIndex])
        stark_dates, stark_n, stark_gains, stark_steps, stark_reps, stark_I_shots, stark_Q_shots, stark_P = stark.load_all()
        stark_p_excited = stark.process_shots(stark_I_shots, stark_Q_shots, stark_n, stark_steps)
        stark_freqs = stark.gain2freq(stark_gains)
    except Exception:
        print("qspec_ge data error or qspec_ge data missing in data-taking block")

if analysis_flags["load_optimization_data"]:
    try:
        print("loading optimization rspec g-e data... ")
        rspec_ge = rspec(data_dir, dataset, QubitIndex, folder="optimization")
        rspec_dates, rspec_n, rspec_probe_freqs, rspec_mags, rspec_freqs, rspec_freq_centers = rspec_ge.load_all()
        start_opt_rspec = rspec_dates[0]
    except Exception:
        print("rspec_ge data error or rspec_ge data missing in optimization block")

    try:
        print("Loading optimization g-e qspec data...")
        opt_qspec_ge = qspec(data_dir, dataset, QubitIndex, folder="optimization")
        opt_qspec_dates, opt_qspec_n, opt_qspec_probe_freqs, opt_qspec_I, opt_qspec_Q = opt_qspec_ge.load_all()
        opt_qspec_freqs, opt_qspec_errs, opt_qspec_fwhms = opt_qspec_ge.get_all_qspec_freq(opt_qspec_probe_freqs, opt_qspec_I, opt_qspec_Q, opt_qspec_n)
        start_opt_qspec = opt_qspec_dates[0]
    except Exception:
        print("qspec_ge data error or qspec_ge data missing in optimization block")

    try:
        print("Loading optimization T1 data...")
        opt_t1_ge = t1(data_dir, dataset, QubitIndex, theta, threshold, folder="optimization")
        opt_t1_dates, opt_t1_n, opt_delay_times, opt_t1_steps, opt_t1_reps, opt_t1_I_shots, opt_t1_Q_shots = opt_t1_ge.load_all()
        opt_t1_p_excited = opt_t1_ge.process_shots(opt_t1_I_shots, opt_t1_Q_shots, opt_t1_n, opt_t1_steps)
    except Exception:
        print("t1_ge data error or t1_ge data missing in optimization block")

    try:
        print("Loading optimization amp_rabi_ge data...")
        amp_rabi_ge = ampRabi(data_dir, dataset, QubitIndex, folder="optimization")
        rabi_dates, rabi_n, rabi_gains, rabi_I, rabi_Q = amp_rabi_ge.load_all()
        pi_amps = amp_rabi_ge.get_all_pi_amp(rabi_gains, rabi_I, rabi_Q, rabi_n)
    except Exception:
        print("amp_rabi_ge data error or amp_rabi_ge data missing in optimization block")

    try:
        print("Loading optimization SSF data...")
        opt_ssf_ge = ssf(data_dir, dataset, QubitIndex, folder="optimization")
        opt_ssf_dates, opt_ssf_n, opt_I_g, opt_Q_g, opt_I_e, opt_Q_e, opt_fid, opt_angles = opt_ssf_ge.load_all()
    except Exception:
        print("ssf_ge data error or ssf_ge data missing in optimization block")

if analysis_flags["timestream"]:
    print("Generating timestream plots...")

    fig, ax = plt.subplots(3,2, layout='constrained')
    fig.suptitle(f'{substudy} dataset {dataset} qubit {QubitIndex + 1} timestream')

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
        plot = ax[0][0]
        try:
            qspec_mag = np.sqrt(np.square(qspec_I)+np.square(qspec_Q))
            qfreq, qfreq_err, fwhm, qspec_fit = qspec_ge.get_qspec_freq_in_round(qspec_probe_freqs, qspec_I, qspec_Q, round, qspec_n, plot=False)

            plot.plot(qspec_probe_freqs, qspec_mag[round], label=f'round {round + 1}')
            plot.plot(qspec_probe_freqs, qspec_fit, 'k:')
            plot.set_xlabel('qubit probe frequency [MHz]',fontsize=sz)
            plot.set_ylabel('I,Q magnitude [a.u.]',fontsize=sz)
            plot.set_title('low gain qspec_ge',fontsize=sz)
            plot.legend()
        except Exception:
            plot.set_title("qspec_ge data error",fontsize=sz)


        ##### high gain qspec #####
        plot = ax[0][1]
        try:
            plot.plot(hgqspec_probe_freqs, hgqspec_mag[round], label=f'round {round + 1}')
            plot.legend()
            plot.set_xlabel('qubit probe frequency [MHz]',fontsize=sz)
            plot.set_ylabel('I,Q magnitude [a.u.]',fontsize=sz)
            plot.set_title('high gain qspec_ge',fontsize=sz)
        except Exception:
            plot.set_title("high gain qspec_ge data error",fontsize=sz)

        ##### t1 data #####
        plot = ax[0][2]

        try:
            q1_fit_exponential, T1_err, T1_est = t1_ge.get_t1_in_round(delay_times, t1_p_excited, t1_n, round, plot = False)

            plot.plot(delay_times, t1_p_excited[round], label=f'round {round + 1} T1 = {T1_est:.2f} +/- {T1_err:.2f} us')
            plot.plot(delay_times, q1_fit_exponential, 'k:')
            plot.set_title('t1_ge',fontsize=sz)
            plot.set_ylabel('P(e)',fontsize=sz)
            plot.set_xlabel('delay time [us]',fontsize=sz)
        except Exception:
            plot.set_title("t1_ge data error",fontsize=sz)

        #resonator stark data#
        plot = ax[1][0]

        try:
            rstark_p_excited_in_round = rstark.get_p_excited_in_round(rstark_gains, rstark_p_excited, rstark_n, round, plot = False)
            plot.plot(rstark_freqs, rstark_p_excited_in_round, label=f'round {round + 1}')
            plot.set_xlabel('stark shift [MHz]',fontsize=sz)
            plot.set_ylabel('P(e)',fontsize=sz)
            plot.set_title('stark tone @ resonator frequency',fontsize=sz)
            plot.legend()
        except Exception:
            plot.set_title("res stark spec data error",fontsize=sz)

        stark_p_excited_in_round = stark.get_p_excited_in_round(stark_gains, stark_p_excited, stark_n, round, plot = False)

        # stark spec data #
        plot = ax[1][1]
        try:
            plot.plot(stark_freqs, stark_p_excited_in_round, label=f'round {round + 1}')
            plot.set_xlabel('stark shift [MHz]')
            plot.set_ylabel('P(e)')
            plot.set_title('stark tone @ fixed detuning')
            plot.legend()
        except Exception:
            plot.set_title("stark spec data error",fontsize=sz)


        # ig_new, qg_new, ie_new, qe_new, xg, yg, xe, ye = ssf_ge.get_ssf_in_round(I_g, Q_g, I_e, Q_e, round)
        #
        # plot = ax[1][2]
        # plot.scatter(ig_new, qg_new, s=2, color='b', label='g')
        # plot.scatter(ie_new, qe_new, s=2, color='r', label='e')
        # plot.set_xlabel('I [a.u.]')
        # plot.set_ylabel('Q [a.u.]')
        # plot.legend()


    fig.suptitle(f'dataset {dataset} qubit {QubitIndex + 1}')

if analysis_flags['g-e_optimization_report']:
    fig, ax = plt.subplots(3,3, layout='constrained', figsize=[12,8])
    fig.suptitle(f"g-e optimization for qubit {QubitIndex + 1} dataset: {dataset} {substudy}", fontsize=14)
    plt.rcParams['font.size'] = 10 #this doesn't work

    plot = ax[0][0]
    try:
        rspec_mag, rspec_freq = rspec_ge.get_rspec_freq_in_round(rspec_freqs, rspec_mags, 0, rspec_n, plot=False)
        plot.plot(np.array(rspec_probe_freqs) + rspec_freq_centers[0], rspec_mag)
        plot.plot([rspec_freq, rspec_freq], [np.min(rspec_mag), np.max(rspec_mag)], 'r:')
        plot.set_xlabel('resonator probe frequency [MHz]',fontsize=sz)
        plot.set_ylabel('I,Q magnitude [a.u.]',fontsize=sz)
        plot.set_title(f'res_ge {np.round(rspec_freq,2)} MHz, round {0 + 1} of {rspec_n}',fontsize=sz)
    except Exception:
        plot.set_title("rspec_ge data error",fontsize=sz)

    plot = ax[0][1]
    try:
        plot.errorbar(get_abs_min(start_opt_rspec, rspec_dates), rspec_freqs, fmt='o')
        plot.set_xlabel('time [min]',fontsize=sz)
        plot.set_ylabel('resonator frequency [MHz]',fontsize=sz)
        plot.set_title('rspec_ge timestream',fontsize=sz)
    except Exception:
        plot.set_title("rspec_ge data error",fontsize=sz)

    ###qspec-ge####
    plot = ax[0][2]
    try:
        opt_qfreq, opt_qfreq_err, opt_qfwhm, opt_qspec_fit = opt_qspec_ge.get_qspec_freq_in_round(opt_qspec_probe_freqs, opt_qspec_I, opt_qspec_Q, 0, opt_qspec_n, plot=False)
        opt_qspec_mag = np.sqrt(np.square(opt_qspec_I) + np.square(opt_qspec_Q))
        plot.plot(opt_qspec_probe_freqs, np.transpose(opt_qspec_mag))
        plot.plot(opt_qspec_probe_freqs, opt_qspec_fit, label='Lorentzian')
        plot.plot([opt_qfreq, opt_qfreq], [np.min(opt_qspec_mag), np.max(opt_qspec_mag)], 'r:')
        plot.legend(loc='best',fontsize=sz)
        plot.set_xlabel('qubit probe frequency [MHz]',fontsize=sz)
        plot.set_ylabel('I,Q magnitude [a.u.]',fontsize=sz)
        plot.set_title(f'qspec_ge: {np.round(opt_qfreq, 2)} +/- {np.round(opt_qfreq_err, 2)} MHz, fwhm: {np.round(opt_qfwhm, 2)} MHz',fontsize=sz)
    except Exception:
        plot.set_title("qspec_ge data error",fontsize=sz)

    plot = ax[1][0]
    try:
        plot.errorbar(get_abs_min(start_opt_qspec, opt_qspec_dates), opt_qspec_freqs, opt_qspec_errs, fmt='o')
        plot.set_xlabel('time [min]',fontsize=sz)
        plot.set_ylabel('qubit frequency [MHz]',fontsize=sz)
        plot.set_title('qspec_ge timestream',fontsize=sz)
    except Exception:
        plot.set_title("qspec_ge data",fontsize=sz)


    ### t1 ####
    plot = ax[1][1]
    try:
        opt_q1_fit_exponential, opt_T1_err, opt_T1_est = opt_t1_ge.get_t1_in_round(opt_delay_times, opt_p_excited, opt_t1_n, 0, plot=True)
        plot.plot(opt_delay_times, opt_t1_p_excited[0])
        plot.plot(opt_delay_times, opt_q1_fit_exponential, 'k:')
        plot.legend(fontsize=sz)
        plot.set_title(f't1_ge: {opt_T1_est:.2f} +/- {opt_T1_err:.2f} us',fontsize=sz)
        plot.set_ylabel('P(e)',fontsize=sz)
        plot.set_xlabel('delay time [us]',fontsize=sz)
    except Exception:
        plot.set_title("t1_ge data error",fontsize=sz)

    ### amp rabi ####
    plot = ax[1][2]
    try:
        pi_amp, cosine_fit, mag = amp_rabi_ge.get_pi_amp_in_round(rabi_gains, rabi_I, rabi_Q, 0)
        plot.plot(rabi_gains, mag)
        plot.plot(rabi_gains, cosine_fit, label='cosine')
        plot.set_title(f'amp_rabi_ge: {pi_amp:.2f} a.u.', fontsize=sz)
        plot.set_ylabel('I,Q magnitude [a.u.]', fontsize=sz)
        plot.set_xlabel('gain [a.u.]', fontsize=sz)
        plot.legend(fontsize=sz)
    except Exception:
        plot.set_title("amp_rabi_ge data error",fontsize=sz)

    ### resonator offset frequency optimization ###
    plot = ax[2][0]
    plot.set_title("coming soon: resonator offset frequency",fontsize=sz)

    #### ssf ######
    plot = ax[2][1]
    try:
        ig_new, qg_new, ie_new, qe_new, xg, yg, xe, ye, theta0, threshold0, fid0 = opt_ssf_ge.get_ssf_in_round(opt_I_g, opt_Q_g, opt_I_e, opt_Q_e, 0)
        plot.scatter(ig_new, qg_new, c='b',label='g',s=2)
        plot.scatter(ie_new, qe_new, c='r',label='e',s=2)
        plot.set_title(f'ssf_ge: theta = {np.round(theta0,3)}')
        plot.set_xlabel('I [a.u.]', fontsize=sz)
        plot.set_ylabel('Q [a.u.]', fontsize=sz)
        plot.plot([threshold0, threshold0],[np.min(qe_new), np.max(qe_new)], 'k:', linewidth=2)
        plot.legend(fontsize=sz)
        plot.set_aspect("equal")
    except Exception as e:
        raise e
        plot.set_title("ssf_ge data error",fontsize=sz)

    plot = ax[2][2]
    try:
        xlims = [np.min(ig_new), np.max(ie_new)]
        ng, binsg,pg = plot.hist(ig_new, bins=100, range=xlims, color='b', label='g', alpha=0.5)
        ne, binse, pe = plot.hist(ie_new, bins=100, range=xlims, color='r', label='e', alpha=0.5)
        plot.plot([threshold0, threshold0], [0, np.max(ne)*1.2], 'k:', linewidth=2)
        plot.set_xlabel('I [a.u.]',fontsize=sz)
        plot.set_title(f'ssf_ge: fidelity = {np.round(fid0,3) *100} %',fontsize=sz)

    except Exception as e:
        raise e
        plot.set_title("ssf_ge data error",fontsize=sz)


plt.show()





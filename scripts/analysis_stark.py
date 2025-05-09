import numpy as np
import matplotlib.pyplot as plt
import os

from qicklab.analysis import rspec, qspec, t1, ssf, resstarkspec, starkspec, auto_threshold, ampRabi
from qicklab.utils import get_abs_min

############### set values here ###################
data_dir = "/Users/joycecs/PycharmProjects/PythonProject/.venv/QUIET/QUIET_data/RR_comprehensive_TLS/" #update based on file transfer location from Ryan
#study_dir = "/exp/cosmiq/data/QUIET/QICK_data/run6/6transmon/TLS_Comprehensive_Study/" #update based on file transfer location from Ryan
substudy = 'source_off_substudy1'
#data_dir = os.path.join(study_dir, substudy)
dataset = '2025-04-15_21-24-46'

QubitIndex = 0 #zero indexed
analysis_flags = {"get_threshold": False, "load_all_data": True, "timestream": False, "round": False, "ssf_rounds": True}
selected_round = [10, 20, 30, 40, 50]
threshold = -920 #overwritten when get_threshold flag is set to True
theta = -2.27 #overwritten when get_threshold flag is set to True
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
        print("Loading SSF data...")
        ssf_ge = ssf(data_dir, dataset, QubitIndex)
        ssf_dates, ssf_n, I_g, Q_g, I_e, Q_e= ssf_ge.load_all()
        thetas, thresholds, fids = ssf_ge.get_all_ssf(I_g, Q_g, I_e, Q_e, ssf_n)
        start_time = ssf_dates[0]
    except Exception:
        print("ssf_ge data error or ssf_ge data missing in data-taking block")

    try:
        print("Loading detuning Stark spec data...")
        stark = starkspec(data_dir, dataset, QubitIndex, duffing_constant[QubitIndex], theta, threshold, anharmonicity[QubitIndex], detuning[QubitIndex])
        stark_dates, stark_n, stark_gains, stark_steps, stark_reps, stark_I_shots, stark_Q_shots, stark_P = stark.load_all()
        stark_p_excited = stark.process_shots(stark_I_shots, stark_Q_shots, stark_n, stark_steps)
        stark_freqs = stark.gain2freq(stark_gains)
    except Exception:
        print("qspec_ge data error or qspec_ge data missing in data-taking block")

if analysis_flags["timestream"]:
    print("Generating timestream plots...")

    fig, ax = plt.subplots(4,1, layout='constrained')
    fig.suptitle(f'{substudy} dataset {dataset} qubit {QubitIndex + 1} timestream')


    ##### single-shot ##########
    plot = ax[0]
    try:
        plot.errorbar(get_abs_min(start_time, ssf_dates), np.array(fids) * 100, fmt='o')
        plot.set_xlabel('time [min]')
        plot.set_ylabel('single-shot fidelity [%]')
        for i in selected_round:
            plot.scatter((ssf_dates[i] - start_time).total_seconds() / 60, fids[i] * 100, marker="o", s=200, alpha=0.5)
    except Exception as e:
        raise e
        plot.set_title("ssf_ge data error")

    plot = ax[1]
    try:
        plot.plot(get_abs_min(start_time, ssf_dates), np.array(thetas), marker='o')
        plot.set_xlabel('time [min]')
        plot.set_ylabel('ssf angle [rad]')
    except Exception as e:
        raise e
        plot.set_title("ssf_ge data error")

    plot = ax[2]
    try:
        plot.plot(get_abs_min(start_time, ssf_dates), np.array(thresholds), marker='o')
        plot.set_xlabel('time [min]')
        plot.set_ylabel('threshold')
    except Exception as e:
        raise e
        plot.set_title("ssf_ge data error")

    #### stark spec at fixed detuning #####
    plot = ax[3]
    try:
        cbar = plt.colorbar(
            plot.pcolormesh(get_abs_min(start_time, stark_dates), stark_freqs, np.transpose(stark_p_excited),
                            shading="nearest", cmap="viridis"), ax=plot)
        cbar.set_label("P(MS=1)")
        plot.set_ylabel('stark shift [MHz]')
        plot.set_xlabel('time [min]')
        plot.set_title('stark tone @ fixed detuning from qubit frequency')
        for i in selected_round:
            plot.scatter((stark_dates[i] - start_time).total_seconds() / 60, stark_freqs[len(stark_freqs) - 1],
                         marker="^", s=150, alpha=1.0)
    except Exception as e:
        raise e
        plot.set_title("stark spec data error")

if analysis_flags["round"]:
    print("Generating individual round plots...")

    for round in selected_round:
        fig, ax = plt.subplots(2, 3, layout='constrained')
        fig.suptitle(f'{substudy} dataset {dataset} qubit {QubitIndex + 1} round {round}')
        # stark spec data #
        try:
            row = 0
            col = 1
            selected_steps = [0, 49, 99, 149, 199]
            for step_idx in selected_steps:
                auto = auto_threshold(data_dir, dataset, QubitIndex)
                I_shots, Q_shots = auto.load_sample(idx=round, step_idx=step_idx)
                theta, threshold, i_new, q_new = auto.get_threshold(I_shots, Q_shots, plot=False)
                state = (i_new < threshold)

                plot = ax[row][col]
                plot.scatter(i_new, q_new, marker="o", c=state)
                plot.plot([threshold, threshold],[np.min(q_new), np.max(q_new)],'k:')
                plot.set_xlabel("I [a.u.]")
                plot.set_ylabel("Q [a.u.]")
                plot.set_title(f"stark shift = {np.round(stark_freqs[step_idx],2)} MHz, theta = {np.round(theta, 3)}, threshold = {np.round(threshold, 2)}", fontsize=sz)
                plot.set_aspect("equal")
                col+=1
                if col > 2:
                    row = 1
                    col = 0

        except Exception as e:
            raise e
            plot.set_title("stark spec data error", fontsize=sz)

        plot = ax[0][0]
        try:
            theta0, threshold0, fid0, ig_new, qg_new, ie_new, qe_new, xg, yg, xe, ye = ssf_ge.get_ssf_in_round(
                I_g, Q_g, I_e, Q_e, round)
            plot.scatter(ig_new, qg_new, c='b', label='g', s=2)
            plot.scatter(ie_new, qe_new, c='r', label='e', s=2)
            plot.set_title(f'ssf_ge: fidelity = {np.round(fid0,3)*100}%, theta = {np.round(theta0, 3)}, threshold = {np.round(threshold0, 2)}', fontsize=sz)
            plot.scatter(xg, yg, c='k', s=6)
            plot.scatter(xe, ye, c='k', s=6)
            plot.set_xlabel('I [a.u.]', fontsize=sz)
            plot.set_ylabel('Q [a.u.]', fontsize=sz)
            plot.plot([threshold0, threshold0], [np.min(qe_new), np.max(qe_new)], 'k:', linewidth=2)
            plot.legend(fontsize=sz)
            plot.set_aspect("equal")
        except Exception:
            plot.set_title("ssf_ge data error", fontsize=sz)

if analysis_flags["ssf_rounds"]:
    print("Generating ssf rounds plots...")
    fig, ax = plt.subplots(2, len(selected_round), layout='constrained')
    fig.suptitle(f'{substudy} dataset {dataset} qubit {QubitIndex + 1}')

    idx = 0
    for round in selected_round:

            plot = ax[0][idx]
            theta0, threshold0, fid0, ig_new, qg_new, ie_new, qe_new, xg, yg, xe, ye = ssf_ge.get_ssf_in_round(
                I_g, Q_g, I_e, Q_e, round)
            plot.scatter(ig_new, qg_new, c='b', label='g', s=2)
            plot.scatter(ie_new, qe_new, c='r', label='e', s=2)
            plot.set_title(f'ssf_ge round {round} threshold = {np.round(threshold0, 2)}', fontsize=sz)
            plot.scatter(xg, yg, c='k', s=6)
            plot.scatter(xe, ye, c='k', s=6)
            plot.set_xlabel('I [a.u.]', fontsize=sz)
            plot.set_ylabel('Q [a.u.]', fontsize=sz)
            plot.plot([threshold0, threshold0], [np.min(qe_new), np.max(qe_new)], 'k:', linewidth=2)
            plot.legend(fontsize=sz)
            plot.set_aspect("equal")

            plot = ax[1][idx]
            xlims = [np.min(ig_new), np.max(ie_new)]
            ng, binsg, pg = plot.hist(ig_new, bins=100, range=xlims, color='b', label='g', alpha=0.5)
            ne, binse, pe = plot.hist(ie_new, bins=100, range=xlims, color='r', label='e', alpha=0.5)
            plot.plot([threshold0, threshold0], [0, np.max(ne) * 1.2], 'k:', linewidth=2)
            plot.set_xlabel('I [a.u.]', fontsize=sz)
            plot.set_title(f'ssf_ge: fidelity = {np.round(fid0, 2) * 100} %', fontsize=sz)

            idx = idx + 1


plt.show()
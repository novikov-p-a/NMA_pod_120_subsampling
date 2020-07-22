import numpy as np
import csv
import matplotlib.pyplot as plt

def tsvout(dir, fname):
    print("=== " + fname + ".tsv: ===")
    with open(dir + fname + ".tsv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")        
        for line in tsvreader:
            print(line[1:])

def npyload(dir, fname):
    print("=== " + fname + ".npy: ===")
    data = np.load(dir + fname + ".npy", allow_pickle=True)
    print("shape: ", data.shape)
    print(data)
    return data

# plot rusters for selected clusters and trials for 1-day 1-mice dataset
def plot_raster_for_clusters_and_trials(
        path="D:\\NMA\\steinmetz_full", # default for: full path to folder with full Steinmetz dataset
        mice_and_date="Cori_2016-12-14", # default for: 1-day 1-mice dataset folder name
        trial_nums = [0,3,6], # default for: array of selected trials
        cluster_nums = range(20) # default for: array of selected clusters
        ):
    dir = path + "\\" + mice_and_date + "\\"

    spikes_times = npyload(dir, "spikes.times")
    spike_clusters = npyload(dir, "spikes.clusters")
    trials_intervals = npyload(dir, "trials.intervals")
    trials_goCue_times = npyload(dir, "trials.goCue_times")
    trials_feedback_times = npyload(dir, "trials.feedback_times")
    trials_visualStim_times = npyload(dir, "trials.visualStim_times")    
   
    # check how many spikes in each cluster
    uncl = np.unique(spike_clusters)
    spikes_each_cluster = np.ndarray(len(uncl))
    times_each_cluster = np.ndarray(len(uncl),dtype=np.ndarray)
    
    i = 0
    loc_uncl = np.ndarray(len(cluster_nums), dtype=int)    
    for icl in uncl:
        if icl in cluster_nums:
            loc_uncl[i] =  icl
            spikes_each_cluster[icl] = np.sum(spike_clusters == icl)
            spikes_icl = spikes_times[spike_clusters == icl]                
            spikes_icl_seltrials = []
            for trial_num in trial_nums:
                spikes_icl_onetrial = spikes_icl[spikes_icl >= trials_intervals[trial_num][0]]
                spikes_icl_onetrial = spikes_icl_onetrial[spikes_icl_onetrial <= trials_intervals[trial_num][1]]
                spikes_icl_seltrials = np.append(spikes_icl_seltrials, spikes_icl_onetrial)
            times_each_cluster[icl] = spikes_icl_seltrials        
            print(times_each_cluster[icl].shape)
            i += 1
            
    # select visual stim, cue and feedback times for selected trials
    stim_vlines = np.ndarray(len(trial_nums)) # visual stim
    cue_vlines = np.ndarray(len(trial_nums)) # goCue
    fb_vlines = np.ndarray(len(trial_nums)) # feedback
    start_vlines = np.ndarray(len(trial_nums)) # trial start
    finish_vlines = np.ndarray(len(trial_nums)) # trial finish
    i = 0
    for trial_num in trial_nums:
        cue_vlines[i] = trials_goCue_times[trial_num]
        fb_vlines[i] = trials_feedback_times[trial_num]
        stim_vlines[i] = trials_visualStim_times[trial_num]
        start_vlines[i] = trials_intervals[trial_num][0]
        finish_vlines[i] = trials_intervals[trial_num][1]
        i += 1
            
    # plot raster
    plt.eventplot(times_each_cluster[loc_uncl], color=".2")
    # plot visual stim, cue and feedback times
    for vline in start_vlines:    
        if vline == start_vlines[0]:
            plt.axvline(vline, color="cyan", label = 'trial boundaries')
        plt.axvline(vline, color="cyan")
    for vline in finish_vlines:        
        plt.axvline(vline, color="cyan")        
    for vline in stim_vlines:
        if vline == stim_vlines[0]:
            plt.axvline(vline, color="red", label = 'visual stimulus')
        plt.axvline(vline, color="red")
    for vline in cue_vlines:
        if vline == cue_vlines[0]:
            plt.axvline(vline, color="orange", label = 'cue')
        plt.axvline(vline, color="orange")
    for vline in fb_vlines:
        if vline == fb_vlines[0]:
            plt.axvline(vline, color="green", label = 'feedback')
        plt.axvline(vline, color="green")
    # tune plot
    plt.xlabel("time, s")
    plt.ylabel("cluster numbers")
    plt.yticks(cluster_nums);
    plt.title(mice_and_date)
    plt.legend()
    plt.show()

plot_raster_for_clusters_and_trials()

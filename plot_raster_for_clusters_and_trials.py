import numpy as np
import csv
import os
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

# from SteinmetzHelpers.py
def get_good_cells(fdirpath):
    # location in brain of each neuron
    brain_loc = os.path.join(fdirpath, "channels.brainLocation.tsv")

    good_cells = (np.load(os.path.join(fdirpath, "clusters._phy_annotation.npy")) >= 2 ).flatten()
    clust_channel = np.load(os.path.join(fdirpath, "clusters.peakChannel.npy")).astype(int) - 1
    br = []
    with open(brain_loc, 'r') as tsv:
        tsvin = csv.reader(tsv, delimiter="\t")
        k=0
        for row in tsvin:
            if k>0:
                br.append(row[-1])
            k+=1
    br = np.array(br)
    good_cells = np.logical_and(good_cells, clust_channel.flatten()<len(br))
    brain_region = br[clust_channel[:,0]]


    return good_cells, brain_region, br

# ROIs:
# Visual areas
VisualROI = ('VISa', 'VISam', 'VISI', 'VISp', 'VISpm', 'VISrl')
# Motor areas
MotorROI = ('MOp', 'MOs')


def plot_raster_for_ROIclusters_and_trials(
        ROIname = "visual", # "visual" or "motor"
        path="D:\\NMA\\steinmetz_full", # default for: full path to folder with full Steinmetz dataset
        mice_and_date="Cori_2016-12-14", # default for: 1-day 1-mice dataset folder name
        trial_nums = [0,2,3,6], # default for: array of selected trials. If [-1] then all trials
        ):
    
    dir = path + "\\" + mice_and_date + "\\"    
        
    channels_probe = npyload(dir, "channels.probe")
    channels_site = npyload(dir, "channels.site")
    
    goodgoodcells, brain_region, br = get_good_cells(dir)
    
    clusters_visual = []
    for ivis in VisualROI:
        matched_clusters = np.where(brain_region == ivis)
        clusters_visual.extend(list(matched_clusters[0]))
        
    clusters_motor = []
    for imot in MotorROI:
        matched_clusters = np.where(brain_region == imot)
        clusters_motor.extend(list(matched_clusters[0]))
        
    spikes_times = npyload(dir, "spikes.times")
    spike_clusters = npyload(dir, "spikes.clusters")
    trials_intervals = npyload(dir, "trials.intervals")
            
    # Retrive spikes from ROIs 
    # Visual ROIs
    nspikes_visual = np.ndarray(len(clusters_visual))
    spikes_visual = np.ndarray(len(clusters_visual),dtype=np.ndarray)
    count = 0    
    for icl in clusters_visual:    
        nspikes_visual[count] = np.sum(spike_clusters == icl)
        spikes_icl = spikes_times[spike_clusters == icl]        
        spikes_visual[count] = spikes_icl
        count += 1
    
    # Motor ROIs
    nspikes_motor = np.ndarray(len(clusters_motor))
    spikes_motor = np.ndarray(len(clusters_motor),dtype=np.ndarray)
    count = 0
    for icl in clusters_motor:    
        nspikes_motor[count] = np.sum(spike_clusters == icl)
        spikes_icl = spikes_times[spike_clusters == icl]        
        spikes_motor[count] = spikes_icl
        count += 1
        
    plot_title = "first 15 clusters of visual"
    clusters_for_plot = clusters_visual[:15]
    if ROIname == "visual":
        clusters_for_plot = clusters_visual
    elif ROIname == "motor":
        clusters_for_plot = clusters_motor
        
    if -1 in trial_nums:
        trial_nums = range(trials_intervals.shape[0])
        
    plot_raster_for_clusters_and_trials(
        path=path,
        mice_and_date=mice_and_date,
        trial_nums = trial_nums,
        cluster_nums =  clusters_for_plot, # [100,101,102,105,106]
        plot_title = ROIname,
        show_each_ytick = False,
        compress_yticks = False,
        show_trials_marks = True
        )

# plot rusters for selected clusters and trials for 1-day 1-mice dataset
def plot_raster_for_clusters_and_trials(
        path="D:\\NMA\\steinmetz_full", # default for: full path to folder with full Steinmetz dataset
        mice_and_date="Cori_2016-12-14", # default for: 1-day 1-mice dataset folder name
        trial_nums = [0,3,6], # default for: array of selected trials
        cluster_nums = range(20), # default for: array of selected clusters
        plot_title = None, # e.g., 'visual', 'motor', etc.
        show_each_ytick = True, # if compress_yticks == True then show_each_ytick is forced to False
        compress_yticks = False, # if compress_yticks == True then show_each_ytick is forced to False
        show_trials_marks = True # plot or do not plot vertical lines for stimuli, cue, feedback etc.
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
    if compress_yticks:
        plt.eventplot(times_each_cluster[loc_uncl], color=".2")
    else:
        plt.eventplot(times_each_cluster[loc_uncl], lineoffsets=cluster_nums, color=".2")
    
    # plot visual stim, cue and feedback times        
    if show_trials_marks:        
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
    plt.ylabel("# cluster")
    if show_each_ytick:
        if compress_yticks == False:
            plt.yticks(cluster_nums);
    if plot_title == None:
        plt.title(mice_and_date)
    else:
        plt.title(mice_and_date + ": " + plot_title)
    plt.legend()
    plt.show()

plot_raster_for_ROIclusters_and_trials(ROIname = "motor")
#plot_raster_for_ROIclusters_and_trials(trial_nums=[-1])


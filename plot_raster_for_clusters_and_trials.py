import numpy as np
import csv
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn import svm

from sklearn.linear_model import LogisticRegression

import sklearn.metrics as skmetrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, auc

def tsvout(dir, fname):
    print("=== " + fname + ".tsv: ===")
    with open(dir + fname + ".tsv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")        
        for line in tsvreader:
            print(line[1:])

def npyload(dir, fname, dataprint=False):
    print("=== " + fname + ".npy: ===")
    data = np.load(dir + fname + ".npy", allow_pickle=True)
    print("shape: ", data.shape)
    if dataprint:
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
        trial_nums = [-1], # [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] # default for: array of selected trials. If [-1] then all trials
        bin_size = 0.02
        ):
    
    dir = path + "\\" + mice_and_date + "\\"
    
    reaction_times = npyload(dir, "reaction_times")
    m = reaction_times[0]
    print(m.shape)
    for rt_elem in m:
        print(rt_elem)
    
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
        
    features, y_all = plot_raster_for_clusters_and_trials(
        path=path,
        mice_and_date=mice_and_date,
        trial_nums = trial_nums,
        cluster_nums =  clusters_for_plot, # [100,101,102,105,106]
        plot_title = ROIname,
        show_each_ytick = False,
        compress_yticks = False,
        show_trials_marks = True,
        bin_size = bin_size
        )
    
    return features, y_all

# plot rusters for selected clusters and trials for 1-day 1-mice dataset
def plot_raster_for_clusters_and_trials(
        path="D:\\NMA\\steinmetz_full", # default for: full path to folder with full Steinmetz dataset
        mice_and_date="Cori_2016-12-14", # default for: 1-day 1-mice dataset folder name
        trial_nums = [0,3,6], # default for: array of selected trials
        cluster_nums = range(20), # default for: array of selected clusters
        plot_title = None, # e.g., 'visual', 'motor', etc.
        show_each_ytick = True, # if compress_yticks == True then show_each_ytick is forced to False
        compress_yticks = False, # if compress_yticks == True then show_each_ytick is forced to False
        show_trials_marks = True, # plot or do not plot vertical lines for stimuli, cue, feedback etc.
        bin_size = 0.02
        ):
    dir = path + "\\" + mice_and_date + "\\"

    spikes_times = npyload(dir, "spikes.times")
    spike_clusters = npyload(dir, "spikes.clusters")
    trials_intervals = npyload(dir, "trials.intervals")
    trials_goCue_times = npyload(dir, "trials.goCue_times")
    trials_feedback_times = npyload(dir, "trials.feedback_times")
    trials_visualStim_times = npyload(dir, "trials.visualStim_times")  
    
    trials_response_choice = npyload(dir, "trials.response_choice")
    trials_response_times = npyload(dir, "trials.response_times")
    trials_visualStim_contrastLeft = npyload(dir, "trials.visualStim_contrastLeft")
    trials_visualStim_contrastRight = npyload(dir, "trials.visualStim_contrastRight")
    
    reaction_times_39m = npyload(dir, "reaction_times")
    reaction_times = []
    for it in reaction_times_39m[0]:        
        reaction_times = np.append(reaction_times, 0.001 * it[0])
   
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
            i += 1
            
    # select visual stim, cue and feedback times for selected trials
    stim_vlines = np.ndarray(len(trial_nums)) # visual stim
    resp_vlines = np.ndarray(len(trial_nums)) # response
    cue_vlines = np.ndarray(len(trial_nums)) # goCue
    fb_vlines = np.ndarray(len(trial_nums)) # feedback
    start_vlines = np.ndarray(len(trial_nums)) # trial start
    finish_vlines = np.ndarray(len(trial_nums)) # trial finish
    rt_vlines = np.ndarray(len(trial_nums)) # visual stim
    i = 0
    for trial_num in trial_nums:
        cue_vlines[i] = trials_goCue_times[trial_num]
        fb_vlines[i] = trials_feedback_times[trial_num]
        stim_vlines[i] = trials_visualStim_times[trial_num]
        resp_vlines[i] = trials_response_times[trial_num]
        rt_vlines[i] = reaction_times[trial_num]
        start_vlines[i] = trials_intervals[trial_num][0]
        finish_vlines[i] = trials_intervals[trial_num][1]
        i += 1
        
    fig, axs = plt.subplots(4, 1)
    fig.patch.set_facecolor('lightgray')
            
    # plot raster
    if compress_yticks:
        axs[1].eventplot(times_each_cluster[loc_uncl], color=".2") # plt.eventplot(times_each_cluster[loc_uncl], color=".2")
    else:
        axs[1].eventplot(times_each_cluster[loc_uncl], lineoffsets=cluster_nums, color=".2") #plt.eventplot(times_each_cluster[loc_uncl], lineoffsets=cluster_nums, color=".2")
        
    # plot visual stim, cue and feedback times        
    if show_trials_marks:        
        for vline in start_vlines:    
            if vline == start_vlines[0]:
                axs[0].axvline(vline, color="lightgray", alpha=0.5, label = 'between trials') # plt.axvline(vline, color="cyan", label = 'trial boundaries')
                axs[1].axvline(vline, color="lightgray", alpha=0.5) # plt.axvline(vline, color="cyan", label = 'trial boundaries')
            axs[0].axvline(vline, color="lightgray", alpha=0.5)
            axs[1].axvline(vline, color="lightgray", alpha=0.5)
        for vline in finish_vlines:        
            axs[0].axvline(vline, color="lightgray", alpha=0.5)        
            axs[1].axvline(vline, color="lightgray", alpha=0.5)
                        
        for i in range(len(start_vlines) - 1):
            axs[0].axvspan(finish_vlines[i], start_vlines[i+1], facecolor='lightgray')
            axs[1].axvspan(finish_vlines[i], start_vlines[i+1], facecolor='lightgray')
                
        for vline in stim_vlines:
            if vline == stim_vlines[0]:
                axs[0].axvline(vline, color="red", label = 'visual stimulus')
                axs[1].axvline(vline, color="red")
            axs[0].axvline(vline, color="red")
            axs[1].axvline(vline, color="red")
        for vline in resp_vlines:
            if vline == resp_vlines[0]:
                axs[0].axvline(vline, color="blue", linestyle='--', dashes=(5, 5), label = 'response')
                axs[1].axvline(vline, color="blue", linestyle='--', dashes=(5, 5))
            axs[0].axvline(vline, color="blue", linestyle='--', dashes=(5, 5))
            axs[1].axvline(vline, color="blue", linestyle='--', dashes=(5, 5))
        iv = 0
        fvl = False
        for vline in rt_vlines:
            if vline == float("inf") or vline == float("-inf"):
                iv = iv
            else:
                iv = iv
                
                if (fvl == False) and (vline == rt_vlines[iv]):
                    axs[0].axvline(vline+stim_vlines[iv], color="black", linestyle='--', dashes=(3, 3), label = 'movement onset')
                    axs[1].axvline(vline+stim_vlines[iv], color="black", linestyle='--', dashes=(3, 3))
                    fvl = True                    
                axs[0].axvline(vline+stim_vlines[iv], color="black", linestyle='--', dashes=(3, 3))
                axs[1].axvline(vline+stim_vlines[iv], color="black", linestyle='--', dashes=(3, 3))
            iv += 1
        for vline in cue_vlines:
            if vline == cue_vlines[0]:
                axs[0].axvline(vline, color="orange", label = 'cue')
                axs[1].axvline(vline, color="orange")
            axs[0].axvline(vline, color="orange")
            axs[1].axvline(vline, color="orange")
        for vline in fb_vlines:
            if vline == fb_vlines[0]:
                axs[0].axvline(vline, color="green", label = 'feedback')
                axs[1].axvline(vline, color="green")
            axs[0].axvline(vline, color="green")
            axs[1].axvline(vline, color="green")
    # tune plot
    axs[1].set_xlabel("time, s")
    axs[1].set_ylabel("# cluster")
    if show_each_ytick:
        if compress_yticks == False:
            axs[1].set_yticks(cluster_nums);
    if plot_title == None:
        axs[1].set_title(mice_and_date)
    else:
        axs[1].set_title(mice_and_date + ": " + plot_title)
    axs[0].legend(loc='upper right')

    # wheel position:
    
    wheel_position = npyload(dir, "wheel.position")
    wheel_timestamps = npyload(dir, "wheel.timestamps")    
    wheelMoves_intervals = npyload(dir, "wheelMoves.intervals")  
    
    trials_intervals_sel = trials_intervals[trial_nums]
    time_min = np.min(trials_intervals_sel)
    time_max = np.max(trials_intervals_sel)
    
    trials_response_choice_sel = trials_response_choice[trial_nums]
        
    wheel_pos_allticks = wheel_timestamps[1][0] - wheel_timestamps[0][0] + 1
    wheel_pos_timestep = 0.01 # in seconds
    wheel_pos_timestart = wheel_timestamps[0][1]
    wheel_pos_timefinish = wheel_timestamps[1][1]
    wheel_pos_alltime = wheel_pos_timefinish - wheel_pos_timestart
    wheel_pos_alltimeticks = int(round(wheel_pos_alltime / wheel_pos_timestep))
    wheel_position_in_time = np.ndarray(wheel_pos_alltimeticks, dtype=np.ndarray)
    timetick = 0
    for t in np.arange(wheel_pos_timestart, wheel_pos_timefinish, step=wheel_pos_timestep):
        wheel_position_tick = int(round(wheel_pos_allticks * (t - wheel_pos_timestart) / wheel_pos_alltime))
        if timetick < wheel_pos_alltimeticks:         
            t_whpos_pair = []
            t_whpos_pair = np.append(t_whpos_pair, t)
            t_whpos_pair = np.append(t_whpos_pair, wheel_position[wheel_position_tick][0])
            wheel_position_in_time[timetick] = t_whpos_pair
        timetick += 1
    
    brush = 10
    brush_mo = 1. / brush
    
    wheel_position_in_time_t = []
    wheel_position_in_time_pos = []
    for pos in wheel_position_in_time[::brush]:
        wheel_position_in_time_t = np.append(wheel_position_in_time_t, pos[0])
        wheel_position_in_time_pos = np.append(wheel_position_in_time_pos, pos[1])
        
    axs[0].axhline(0.0, color="black")
    
    wheel_pos_alltrials_t = np.ndarray(len(trial_nums),dtype=np.ndarray)
    wheel_pos_alltrials_pos = np.ndarray(len(trial_nums),dtype=np.ndarray)
    
    gtrial_intervals = np.ndarray(1,dtype=np.ndarray)
    
    bin_start = -0.1 # in seconds
    bin_finish = +0.100 # in seconds
    
    wheel_position_dirs = np.empty(len(trials_intervals_sel))
    wheel_position_dirs_sign = np.empty(len(trials_intervals_sel))
      
    gt_i = 0
    i = 0
    for interval in trials_intervals_sel:
        timetick1 =  int(round(brush_mo * (interval[0] - wheel_pos_timestart) / wheel_pos_timestep))
        timetick2 =  int(round(brush_mo * (interval[1] - wheel_pos_timestart) / wheel_pos_timestep))
        
        wheel_position_in_interval_t = wheel_position_in_time_t[(timetick1):timetick2]
        wheel_position_in_interval_pos = wheel_position_in_time_pos[(timetick1):timetick2]
        wheel_position_in_interval_pos = wheel_position_in_interval_pos - wheel_position_in_interval_pos[0]        
                
        wheel_pos_alltrials_pos[i] = wheel_position_in_interval_pos
        
        #wheel_pos_alltrials_t[i] = wheel_position_in_interval_t - stim_vlines[i]        
        #wheel_pos_alltrials_t[i] = wheel_position_in_interval_t - cue_vlines[i]        
        wheel_position_in_interval_t_locked = wheel_position_in_interval_t - stim_vlines[i]        
        if rt_vlines[i] == float("inf") or rt_vlines[i] == float("-inf"):
            i = i
        else:
            # Trial selection criteria:
            #if rt_vlines[i] >= 0.125 and rt_vlines[i] <= 0.400: # only potentially "pure" motor reaction on visual stimuli
            if rt_vlines[i] >= 0.0 and rt_vlines[i] <= 1.400 and trials_response_choice_sel[i] != 0:
                
                wheel_pos_alltrials_t[i] = wheel_position_in_interval_t_locked - rt_vlines[i]        
                axs[0].plot(wheel_position_in_interval_t, wheel_position_in_interval_pos, "magenta")  #plt.plot(wheel_position_in_interval_t, wheel_position_in_interval_pos, "magenta")
                axs[2].plot(wheel_pos_alltrials_t[i], wheel_pos_alltrials_pos[i])  #plt.plot(wheel_position_in_interval_t, wheel_position_in_interval_pos, "magenta")
                axs[2].axvline(-rt_vlines[i], color="red")
                gtrial_intervals_1 = np.ndarray(1,dtype=np.ndarray)
                gtrial_interval = []
                gtrial_interval = np.append(gtrial_interval, stim_vlines[i] + rt_vlines[i] + bin_start)
                gtrial_interval = np.append(gtrial_interval, stim_vlines[i] + rt_vlines[i] + bin_finish)
                gtrial_intervals_1[0] = gtrial_interval
                if gt_i == 0:
                    gtrial_intervals = gtrial_intervals_1
                else:
                    gtrial_intervals = np.append(gtrial_intervals, gtrial_intervals_1)
                                        
                timetick_RT =  int(round(brush_mo * (stim_vlines[i] + rt_vlines[i] - wheel_pos_timestart) / wheel_pos_timestep))
                timetick_RT2 =  int(round(brush_mo * (stim_vlines[i] + rt_vlines[i] + 0.100 - wheel_pos_timestart) / wheel_pos_timestep))
                wheel_position_dirs[gt_i] = wheel_position_in_time_pos[timetick_RT2] - wheel_position_in_time_pos[timetick_RT]
                if wheel_position_dirs[gt_i] >= 0:
                    wheel_position_dirs_sign[gt_i] = 1
                else:
                    wheel_position_dirs_sign[gt_i] = -1
                                        
                gt_i += 1

        i += 1
      
    #########################
    
    spikes_each_cluster_f = np.ndarray(len(uncl))
    times_each_cluster_full = np.ndarray(len(uncl),dtype=np.ndarray)
    
    times_each_cluster_by_gt = np.ndarray(len(gtrial_intervals), dtype=np.ndarray)
    for ii in range(len(times_each_cluster_by_gt)):
        temp_array = np.ndarray(len(uncl),dtype=np.ndarray)
        times_each_cluster_by_gt[ii] = temp_array
        
    nbin = (int)(round((bin_finish - bin_start) / bin_size))
    
    # feature matrix -> bins -> clusters (FR for each cluster)
    features = np.ndarray(len(gtrial_intervals), dtype=np.ndarray)
    for ii in range(len(features)):
        temp_bins = np.ndarray(nbin,dtype=np.ndarray)        
        features[ii] = temp_bins
        for jj in range(len(features[ii])):
            temp_clusters = np.ndarray(len(cluster_nums),dtype=np.ndarray)
            features[ii][jj] = temp_clusters
        
    i = 0
    loc_uncl_f = np.ndarray(len(cluster_nums), dtype=int)    
    for icl in uncl:
        if icl in cluster_nums:
            loc_uncl_f[i] =  icl
            spikes_each_cluster_f[icl] = np.sum(spike_clusters == icl)
            spikes_icl_f = spikes_times[spike_clusters == icl]                
            spikes_icl_seltrials = []
            ii = 0
            for it in gtrial_intervals:
                spikes_icl_onetrial = spikes_icl_f[spikes_icl_f >= it[0]]
                spikes_icl_onetrial = spikes_icl_onetrial[spikes_icl_onetrial <= it[1]]
                spikes_icl_seltrials = np.append(spikes_icl_seltrials, spikes_icl_onetrial)
                #print("gtrial_intervals:", it )
                times_each_cluster_by_gt[ii][icl] = spikes_icl_onetrial
                
                for ibin in range(nbin):
                    t_start =  it[0] + ibin * bin_size
                    t_finish =  it[0] + (ibin + 1) * bin_size
                    spikes_icl_onebin = spikes_icl_f[spikes_icl_f >= t_start]
                    spikes_icl_onebin = spikes_icl_onebin[spikes_icl_onebin < t_finish]
                    nos = len(spikes_icl_onebin)
                    features[ii][ibin][i] = nos
                
                ii += 1
            times_each_cluster_full[icl] = spikes_icl_seltrials        
            
            i += 1            
            
           
    nall = len(features)   
    
    y_all = np.empty(nall)
    for i in range(nall):
        y_all[i] = wheel_position_dirs_sign[i]
        
    print("y_all: ", y_all)
    
    print("times_each_cluster_by_gt.shape:", times_each_cluster_by_gt.shape)
    
    for i in range(len(times_each_cluster_by_gt)):
        if compress_yticks:
            axs[3].eventplot(times_each_cluster_by_gt[i][loc_uncl_f], color=".2") # plt.eventplot(times_each_cluster[loc_uncl], color=".2")
        else:
            axs[3].eventplot(times_each_cluster_by_gt[i][loc_uncl_f], lineoffsets=cluster_nums, color=".2") #plt.eventplot(times_each_cluster[loc_uncl], lineoffsets=cluster_nums, color=".2")
                
        
    axs[2].set_xlabel("time, s") #plt.xlabel("time, s")
    axs[2].set_title(mice_and_date + ": wheel position, all trials")

    axs[2].axvline(0., color="black", label = 'movement onset', linestyle='--', dashes=(3, 3))         
    #axs[2].axvline(0., color="red",  label = 'visual stimulus')
    #axs[2].axvline(0., color="orange", label = 'cue')         
    
    axs[2].set_xlim((-0.5, 2.0)) #plt.ylabel("wheel position")
    
    axs[2].legend()
        
    axs[0].set_xlabel("time, s") #plt.xlabel("time, s")
    axs[0].set_ylabel("wheel position") #plt.ylabel("wheel position")
    axs[0].set_title(mice_and_date)
    
    axs[0].set_xlim((start_vlines[0], finish_vlines[-1])) #plt.ylabel("wheel position")
    axs[1].set_xlim((start_vlines[0], finish_vlines[-1])) #plt.ylabel("wheel position")

    fig.tight_layout()
    plt.show()
    
    return features, y_all

def mypredict(features, y_all):
    
    nall = len(features)
    #ntrain = (int)(round(0.85 * nall))
    #ntest = nall - ntrain 
    
    nbin = features[0].shape[0]
    ncluster = features[0][0].shape[0]
    
    # create X_all from selected features
    size_1d_all = nall * nbin * ncluster
    X_all = np.full(size_1d_all, 666.6).reshape(nall,-1)
       
    cx = 0
    for i in range(nall):
        #print("range(nall)", i)
        for j in range(nbin):
            for k in range(ncluster):                
                X_all[i, j * ncluster + k] = features[i][j][k]
   
                                
    # X and y -> both to test and train
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.15)
          
    # Model and fit
    LogRegModel = LogisticRegression(penalty='l2', max_iter=100)     
    LogRegModel.fit(X_train, y_train)
    
    # Predict
    predicted = LogRegModel.predict(X_test)
    print("predicted: ", predicted)
    print(f'Accuracy of prediction {skmetrics.accuracy_score(y_test, predicted):.1%}')
    print(skmetrics.classification_report(y_test, predicted))
    
    fig, axs = plt.subplots(2, 1)
    
    # X-validation
    acc = cross_val_score(LogRegModel, X_all, y=y_all, cv=10)
    
    f, axs = plt.subplots()
    axs.boxplot(acc, vert=False)
    axs.scatter(acc, np.ones(10))
    axs.set(
      xlabel="Accuracy",
      yticks=[],
      title=f"Average test accuracy: {acc.mean():.1%}"
    )
    
    fig.tight_layout()
    plt.show()
    
    return acc
    
    
def dopredict(ROIname = "motor",
              bin_size = 0.02):
    features, y_all = plot_raster_for_ROIclusters_and_trials(ROIname = ROIname,
                                                             bin_size = bin_size)
    accuracies = mypredict(features, y_all)
    
    return accuracies
 
bin_sizes = np.arange(0.01, 0.03, 0.01)

accs = np.ndarray(len(bin_sizes),dtype=np.ndarray)
for i in range(len(bin_sizes)):    
    acc = dopredict(ROIname = "motor", 
                           bin_size = bin_sizes[i])
    accs[i] = acc
    
print("bin_sizes: ", bin_sizes)
print("len(bin_sizes):", len(bin_sizes))
fig1, axs1 = plt.subplots(len(bin_sizes),1)
for i in range(len(bin_sizes)):
    
    print("accuracies[i]: ", accs[i])
    
    axs1[i].boxplot(accs[i], vert=False, widths=0.6)
    axs1[i].scatter(accs[i], np.ones(len(accs[i])))
    axs1[i].set(
      xlabel="Accuracy",
      xlim=([0.5, 1.0]),
      yticks=[],
      title=f"Average test accuracy: {accs[i].mean():.1%}" + ", bin " + str(1000.0*bin_sizes[i]) + " ms"
    )
    
fig1.tight_layout()
plt.show()

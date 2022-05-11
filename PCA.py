#%%
from urllib import response
from sklearn.decomposition import PCA
#for interactive plotting outside notebook, uncomment the below statement
#%matplotlib qt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#%%
def PCA_reduction(sample, no_electrodes, sampling_rate, ex_var=0.9975, visual=0):
    #el1 = sample[0,:]
    #el2 = sample[1,:]
    #el3 = sample[2,:]
    #X = np.c_[el1, el2, el3]
    time = np.linspace(0,len(sample[0])/sampling_rate,len(sample[0]))

    #reshaping into (samples, no_electrodes) shape suitable for pca.fit()
    X = np.stack(sample, axis=-1)

    pca = PCA() #n_components=
    pca.fit(X)
    #principal axes
    PA = pca.components_.T*100
    #variance explained by each pc
    var = pca.explained_variance_ratio_

    #finding the no. of PCs necessary to explain given ratio of variance
    no_pc = 0
    sum = 0
    while sum<ex_var:
        sum += var[no_pc]
        no_pc += 1

    if visual==1 or visual==2:
        print('Principal Axes:')
        print(PA[:3])
        print('No. of PCs explaining {}% of variance: '.format(ex_var*100), no_pc)

    if visual==2:
        '''
        Plotting 3D sample of data
        '''
        #Only conducting PCA on 3 displayable dimensions for visualisation purposes
        X_d = np.stack(sample[:3], axis=-1)
        pca_d = PCA()
        pca_d.fit(X_d)
        #principal axes
        PA_d = pca_d.components_.T*100

        fig = plt.figure(figsize=(10,7))
        plt.plot(time, np.swapaxes(sample,1,0), color='#52AD89')
        #color='#75BDA1'
        #color='#7CCEAE'
        plt.title('EMG Channels')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (\u03BCV)')
        plt.show()
        fig = plt.figure(figsize=(10, 7))
        #perspective angles of plot
        elev = 30
        azim = 60
        ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=elev, azim=azim)
        ax.scatter(sample[0,:],sample[1,:],sample[2,:], marker="+", color='#52AD89', alpha=0.8)
        ax.set_xlabel('Electrode 0 (\u03BCV)')
        ax.set_ylabel('Electrode 1 (\u03BCV)')
        ax.set_zlabel('Electrode 2 (\u03BCV)')
        ax.set_title('EMG Channels Feature Space')
        soa = np.array([[0,0,0,PA_d[0][0],PA_d[0][1],PA_d[0][2]],[0,0,0,PA_d[1][0],PA_d[1][1],PA_d[1][2]],[0,0,0,PA_d[2][0],PA_d[2][1],PA_d[2][2]]])
        Xs, Y, Z, U, V, W = zip(*soa)
        ax.quiver(Xs, Y, Z, U, V, W, color='#AD5276', label='Principal Directions')
        ax.legend()


        if False:
            #if wishing to plot principal component planes
            pca = PCA(n_components=3)
            pca.fit(Y)
            V = pca.components_.T

            x_pca_axis, y_pca_axis, z_pca_axis = 3 * V
            x_pca_plane = np.r_[x_pca_axis[:2], -x_pca_axis[1::-1]]
            y_pca_plane = np.r_[y_pca_axis[:2], -y_pca_axis[1::-1]]
            z_pca_plane = np.r_[z_pca_axis[:2], -z_pca_axis[1::-1]]
            x_pca_plane.shape = (2, 2)
            y_pca_plane.shape = (2, 2)
            z_pca_plane.shape = (2, 2)
            ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane)
            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
        plt.show()

    '''
    Transforming Data to Principal Components
    '''
    #Data in the in the principal coordinate frame
    X_new = pca.transform(X)
    pc = [[X_new[t][p] for t in range(len(X_new))] for p in range(no_electrodes)]
    if visual==1 or visual==2:
        #ORIGINAL CHANNELS AND 3 PRINCIPAL COMPONENTS
        fig = plt.figure(figsize=(10,7))
        plt.plot(time, np.swapaxes(sample[1:],1,0), color='#52AD89')
        plt.plot(time, sample[0], color='#52AD89', label='channels')
        plt.plot(time, pc[0], color='#AD5276', label='first 3 principal components')
        plt.plot(time, pc[1], color='#AD5276')
        plt.plot(time, pc[2], color='#AD5276')
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (\u03BCV)')
        plt.title('PCA decomposition')
        plt.show()

    '''
    Correlation Matrix
    '''
    from scipy.stats import pearsonr

    #pearson correlation matrix between each electrode channel and each principal component
    correlations = np.zeros((no_electrodes, no_electrodes))
    for e in range(no_electrodes):
        correlations[e,:] = np.array([pearsonr(sample[e,:],pc[p])[0] for p in range(no_electrodes)])

    if visual==1 or visual==2:
        #CORRELATION MATRIX
        fig, ax = plt.subplots(figsize=(7,5))
        im = ax.imshow(correlations, cmap='gray')
        ax.set_xlabel('Principal Components')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Channels')
        ax.set_xticks([t for t in range(no_electrodes)])
        ax.xaxis.tick_top()
        ax.set_yticks([t for t in range(no_electrodes)])
        fig.colorbar(im, label='Pearson coeff.')
        im.set_clim(-1, 1)
        ax.set_title('Correlation Matrix')
        fig.tight_layout()
        plt.show()

        #SORTED CORRELATION MATRIX
        sorted = np.sort(correlations, axis=0)[::-1]
        fig, ax = plt.subplots(figsize=(7,5))
        im = ax.imshow(sorted, cmap='gray')
        ax.set_xlabel('Principal Components')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Correlation Rank (Channels)')
        ax.set_xticks([t for t in range(no_electrodes)])
        ax.xaxis.tick_top()
        ax.set_yticks([t for t in range(no_electrodes)])
        ax.set_yticklabels([t+1 for t in range(no_electrodes)])
        fig.colorbar(im, label='Pearson coeff.')
        im.set_clim(-1, 1)
        ax.set_title('Sorted Correlation Matrix')
        fig.tight_layout()
        plt.show()

    '''
    Finding most influential ('principal') electrodes
    '''
    idx = (-correlations[:,:no_pc]).argsort(axis=0)[:2] #get indices of 2 highest coefficients (first row gives indeces of elec. that is most correlated with each PC. second row gives indeces of second most correlated electrodes)
    idx.flatten() #covnert to 1D array since irrespective if most or 2nd to most correlated
    idx.sort()
    pc_electrodes = np.unique(idx) #remove repeating electrode indeces
    if visual==1 or visual==2:
        print('Most influential electrodes:')
        print(pc_electrodes)
    
    return pc_electrodes


#%%
#histogram
def pc_histogram(data, no_electrodes, title):
    bins_custom = [b for b in range(no_electrodes+1)]
    fig = plt.figure(figsize=(10,7))
    plt.subplot()
    n, bins, patches = plt.hist(x=data, bins=bins_custom,   color='#52AD89',
                                alpha=0.7, rwidth=0.85)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Electrode ID [dimensionless]', fontname="Cambria", fontsize=12)
    plt.ylabel('Count [dimensionless]',fontname="Cambria", fontsize=12)
    plt.title(title,fontname="Cambria", fontsize=12)
    plt.xticks([t+0.5 for t in range(no_electrodes)],[t for t in range(no_electrodes)])
    #plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    #save
    # fig.savefig('pc_electrodes_hist_global.svg', format='svg', dpi=1200)

    plt.show()
# %%
'''
Offline PCA analysis
    - Run PCA for all samples and aggregate results to display global trends in the dataset:
        - which electrodes occur most often as principal electrodes
        - how many principal electrodes are needed 
    - Depends on the var_ratio that we wish to explain
'''
import Load_Data
import Support_Functions
import importlib
importlib.reload(Load_Data)
importlib.reload(Support_Functions)
from Load_Data import *
from Support_Functions import *

'''
Parameters
'''
no_electrodes = 12
sampling_rate = 2000 #Hz
classes = [c+1 for c in range(17)] #which movements to classify based on ID
subjects = [1,2] #subjects to extract
#explained variance
ex_var = 0.9


#stores the histograms for all classes
histograms = np.zeros((len(classes)*len(subjects),no_electrodes))
#stores the principal electrodes for all subjects and classes
global_pc_electrodes = []
#stores the no. of principal electrodes for each sample for each class (key=class ID, value=list of no. of pc_el for each sample)
no_pce = {}
for c in classes:
    no_pce[str(c)] = []
#stores the class ID in the order as the samples come in
y = []
#stores the no. of pc electrodes in the order as the samples come in
no_pce_raw = []
#no. of repetitions per class 
no_reps = 6
for s in subjects:
    emg_labelled, y_temp, time_pose, _, _, _, _ = load_data([s], classes, sampling_rate, no_electrodes)
    y += y_temp
    for c in classes:
        pc_electrodes = []
        for rep in range(no_reps):
            #shape (12, samples)
            #convert to microVolts
            emg_data = emg_labelled[(c-1)*no_reps+rep]*1000000
            emg_data = np.swapaxes(emg_data, 0, 1)
            pce = PCA_reduction(emg_data, no_electrodes, sampling_rate, ex_var=ex_var, visual=0)
            #store no_pce of current sample
            no_pce[str(c)] += [len(pce)]
            no_pce_raw += [len(pce)]
            pc_electrodes += np.ndarray.tolist(pce)

        #histogram of pc electrodes for current class
        #pc_histogram(pc_electrodes, no_electrodes, 'Principal Electrodes Histogram: Class {}'.format(c))
        histograms[len(classes)*(s-1)+c-1] = np.histogram(pc_electrodes, bins=[b for b in range(no_electrodes+1)])[0]
        global_pc_electrodes += pc_electrodes
#histogram of all samples
global_hist = np.sum(histograms, axis=0)
#histogram of all pc electrodes
pc_histogram(global_pc_electrodes, no_electrodes, 'Global Principal Electrodes Histogram')

no_pc_avg = [np.mean(no_pce[str(key+1)]) for key in range(len(classes))]
no_pc_std = [np.std(no_pce[str(key+1)]) for key in range(len(classes))]

#plot no. of principal electrodes vs class over all samples with avaraged reps per class and std.
fig = plt.figure(figsize=(10,7))
plt.scatter(y, no_pce_raw, color='#52AD89', marker='+')
plt.plot(classes, no_pc_avg,color='#AD5276', label='mean')
plt.plot(classes, np.array(no_pc_avg)+np.array(no_pc_std), color='#AD5276',linestyle='dashed', label='std')
plt.plot(classes, np.array(no_pc_avg)-np.array(no_pc_std), color='#AD5276', linestyle='dashed')
plt.xlabel('Class ID [dimensionless]', fontname="Cambria", fontsize=12)
plt.ylabel('Principal Electrodes Count [dimensionless]',fontname="Cambria", fontsize=12)
plt.title('No. of Principal Electrodes',fontname="Cambria", fontsize=12)
plt.xlim(0,len(classes)+1)

#print the indeces of the globally most influential electrodes
no_pce_global = int(np.round(np.average(no_pce_raw)))
pce_global = (-global_hist).argsort()[:no_pce_global]
print('Global Principal Electrodes for ex_var={}%:\n'.format(ex_var*100), pce_global)


# %%
#box plots
data = [no_pce[str(key+1)] for key in range(len(classes))]
fig = plt.figure(figsize =(10, 7))
# Creating plot
bp = plt.boxplot(data, positions=[p+1 for p in range(len(classes))], widths=0.3)
# plt.yticks(np.arange(0,12,1))
markers = ['D','s','D','s','D','s','D','s','D','s','D','s','D','s','D','s','D']
range_c = [p+1 for p in range(len(classes))]
for bin, range_c, m in zip(data, range_c, markers):
    points = range_c*np.ones(len(bin))+np.random.normal(loc=0.0, scale=0.1, size=len(bin))
    plt.scatter(points, bin, alpha=0.4, marker=m, color='#52AD89')

plt.scatter([], [], alpha=0.4, marker=m, color='#52AD89', label='samples')
plt.xlabel('Class ID [dimensionless]',fontname="Cambria", fontsize=12)
plt.ylabel('No. of Principal Electrodes [dimensionless]',fontname="Cambria", fontsize=12)
plt.hlines(np.average(no_pce_raw),0.75, max(y)+0.25,color='#AD5276', label='mean', linestyles='dashed')
plt.legend()
# fig.savefig('global_pce_box_plots.svg', format='svg', dpi=1200)
plt.show()
# %%

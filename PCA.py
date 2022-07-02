'''
Functions needed for the PCA analysis
'''
#%%
from urllib import response
from sklearn.decomposition import PCA
#for interactive plotting outside notebook, uncomment the below statement
#%matplotlib qt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rc('font',family='Palatino')
import numpy as np

#%%
def PCA_reduction(sample, no_electrodes, sampling_rate, ex_var=0.9975, visual=0):
    '''
    Input:
        - sample: array of time-series data of shape (no_electrodes, no_time_samples)
        - no_electrodes: no. of data channels (i.e., dimensions/axes)
        - sampling rate of data sample
        - ex_var: desired variance to be explained by the output principal components
        - visual=0: no visual output
        - visual=1:some visual output (text and correlation matrices)
        - visual=2: all visual output (sample 3D points)
    Output:
        - principal electrodes (axes) that explain the desired variance (ex_var)
    '''
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
        plt.plot(time, np.swapaxes(sample,1,0), color='#04c8e0')
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
        ax.scatter(sample[0,:500],sample[1,:500],sample[2,:500], marker="+", color='#04c8e0', alpha=0.8)
        ax.set_xlabel('Electrode 0 [\u03BCV]')
        ax.set_ylabel('Electrode 1 [\u03BCV]')
        ax.set_zlabel('Electrode 2 [\u03BCV]')
        ax.set_title('EMG Channels Feature Space', fontsize=14)
        soa = np.array([[0,0,0,PA_d[0][0],PA_d[0][1],PA_d[0][2]],[0,0,0,PA_d[1][0],PA_d[1][1],PA_d[1][2]],[0,0,0,PA_d[2][0],PA_d[2][1],PA_d[2][2]]])
        Xs, Y, Z, U, V, W = zip(*soa)
        ax.quiver(Xs, Y, Z, U, V, W, color='#eb0962', label='Principal Directions')
        ax.legend()

        #Only conducting PCA on 2 displayable dimensions for visualisation purposes
        X_d = np.stack(sample[:2], axis=-1)
        pca_d = PCA()
        pca_d.fit(X_d)
        #principal axes
        PA_d = pca_d.components_.T*100
        fig, ax = plt.subplots(figsize=(10, 7))
        #perspective angles of plot
        ax.scatter(sample[0,:500],sample[1,:500], marker="+", color='#04c8e0', alpha=0.5, label='samples')
        ax.set_xlabel('Electrode 0 [\u03BCV]')
        ax.set_ylabel('Electrode 1 [\u03BCV]')
        ax.set_title('EMG Channels Feature Space', fontsize=14)
        soa = np.array([[0,0,PA_d[0][0],PA_d[0][1]],[0,0,PA_d[1][0],PA_d[1][1]]])
        Xs, Y, U, V = zip(*soa)
        ax.quiver(Xs, Y, U, V, color='#eb0962', scale=700, width=0.004)
        ax.plot([],[], color='#eb0962', label='principal directions')
        ax.legend()
        #fig.savefig('Figures/pc_space.png', format='png', dpi=800)


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
        plt.plot(time, np.swapaxes(sample[1:],1,0), color='#04c8e0')
        plt.plot(time, sample[0], color='#04c8e0', label='channels')
        plt.plot(time, pc[0], color='#eb0962', label='first 3 principal components')
        plt.plot(time, pc[1], color='#eb0962')
        plt.plot(time, pc[2], color='#eb0962')
        plt.legend()
        plt.xlabel('Time [s]', fontsize=12)
        plt.ylabel('Voltage [\u03BCV]', fontsize=12)
        plt.title('PCA decomposition', fontsize=14)
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
        ax.set_xlabel('Principal Components', fontsize=12)
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Channels', fontsize=12)
        ax.set_xticks([t for t in range(no_electrodes)])
        ax.xaxis.tick_top()
        ax.set_yticks([t for t in range(no_electrodes)])
        fig.colorbar(im, label='Pearson coeff.')
        im.set_clim(-1, 1)
        ax.set_title('Correlation Matrix', fontsize=14)
        fig.tight_layout()
        #fig.savefig('Figures/corr_matrix.png', format='png', dpi=800)
        plt.show()

        #SORTED CORRELATION MATRIX
        sorted = np.sort(correlations, axis=0)[::-1]
        fig, ax = plt.subplots(figsize=(7,5))
        im = ax.imshow(sorted, cmap='gray')
        ax.set_xlabel('Principal Components', fontsize=12)
        ax.xaxis.set_label_position('top')
        ax.set_ylabel('Correlation Rank (Channels)', fontsize=12)
        ax.set_xticks([t for t in range(no_electrodes)])
        ax.xaxis.tick_top()
        ax.set_yticks([t for t in range(no_electrodes)])
        ax.set_yticklabels([t+1 for t in range(no_electrodes)])
        fig.colorbar(im, label='Pearson coeff.')
        im.set_clim(-1, 1)
        ax.set_title('Sorted Correlation Matrix', fontsize=14)
        fig.tight_layout()
        #fig.savefig('Figures/sorted_corr_matrix.png', format='png', dpi=800)
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

#histogram
def pc_histogram(data, no_electrodes, title):
    '''
    Histogram of how often is each electrode a principal electrode
    '''
    bins_custom = [b for b in range(no_electrodes+1)]
    fig = plt.figure(figsize=(10,7))
    plt.subplot()
    n, bins, patches = plt.hist(x=data, bins=bins_custom,   color='#04c8e0',
                                alpha=0.7, rwidth=0.85)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Electrode ID [dimensionless]', fontname="Palatino", fontsize=12)
    plt.ylabel('Count [dimensionless]',fontname="Palatino", fontsize=12)
    plt.title(title,fontname="Palatino", fontsize=14)
    plt.xticks([t+0.5 for t in range(no_electrodes)],[t for t in range(no_electrodes)])
    #plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    if 'Class 6' in title:
        fig.savefig('Figures/pc_electrodes_class_hist_2.png', format='png', dpi=800)

    plt.show()
# %%

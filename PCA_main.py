'''
Offline PCA analysis
    - Run PCA for all samples (all subjects and all classes of movement) and aggregate results to display global trends in the dataset:
        - which electrodes occur most often as principal electrodes (histogram)
        - how many principal electrodes are needed (box plot)
    - Depends on the var_ratio that we wish to explain
'''
#%%
import importlib
import Load_Data
importlib.reload(Load_Data)
from Load_Data import *
import Support_Functions
importlib.reload(Support_Functions)
from Support_Functions import *
import PCA
importlib.reload(PCA)
from PCA import *

#%%
'''
Parameters
'''
no_electrodes = 12
sampling_rate = 2000 #Hz
classes = [c+1 for c in range(17)] #which movements to classify based on ID in dataset
subjects = [1,2] #subjects to extract
#explained variance
ex_var = 0.9
#no. of repetitions per class (movement)
no_reps = 6

'''
Analysis
'''

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

for s in subjects:
    emg_labelled, y_temp, time_pose, _, _, _, _,_ = load_data([s], classes, sampling_rate, no_electrodes)
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
'''
Box plots
'''
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

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import scipy.io as sio
import matplotlib.patches as patches
from sklearn import preprocessing
import matplotlib.font_manager

mat_contents = sio.loadmat('../data/HarmonicScores.mat')



HarmonicScores = mat_contents['X']
HarmonicScores = preprocessing.scale(HarmonicScores)
bounds = np.vstack([np.min(HarmonicScores,axis=0),np.max(HarmonicScores,axis=0)])
bound_size = bounds[1,:]-bounds[0,:]
bound_envelope = bound_size*0.15



Observed = HarmonicScores[-1,:]
HarmonicScores = np.delete(HarmonicScores,(-1),axis=0)

nu=0.01

clf = svm.OneClassSVM(nu=nu, kernel="rbf",gamma=0.2)
clf.fit(HarmonicScores)

is_outlier = clf.predict(Observed.reshape(1, -1))



Row = 0
Col = 0
f, axarr = plt.subplots(nrows=1,ncols=1, sharey=False,\
                facecolor='white', figsize=(14, 14),squeeze=False)

    
mesh_min = bounds[0,:] - bound_envelope
mesh_max = bounds[1,:] + bound_envelope

xx, yy  = np.meshgrid(np.linspace(mesh_min[0], mesh_max[0]),\
    np.linspace(mesh_min[1], mesh_max[1]))
 
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# drawing the function
cax = axarr[Row,Col].contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 8),
                         cmap=plt.cm.Blues_r)

# Draw boundary    
a = axarr[Row,Col].contour(xx, yy, Z, levels=[0,Z.max()], linewidths=4,\
    colors='red')
frontier = a.collections[0]
frontier.set_label('Learned Boundary')


axarr[Row,Col].contourf(xx, yy, Z, levels=[0, Z.max()], \
    colors='orange')

axarr[Row,Col].scatter(HarmonicScores[:,0],\
    HarmonicScores[:,1],c='white',s=80,\
    label='Prior Models')


fontsize = 32

axarr[Row,Col].scatter(Observed[0],Observed[1],s=500, label='Observed',zorder=10,\
                        marker='*',linewidth='3',edgecolor='r',color='r')
axarr[Row,Col].legend(scatterpoints=1,loc="lower right",\
                        prop=matplotlib.font_manager.FontProperties(size=fontsize))
axarr[Row,Col].set_xlim(mesh_min[0],mesh_max[0])
axarr[Row,Col].set_ylim(mesh_min[1],mesh_max[1])


axarr[Row,Col].tick_params(axis='both', which='major', labelsize=fontsize)
cb = plt.colorbar(cax)
cb.set_label('Distance To Boundary',fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize-2) 
plt.xlabel('First Harmonic Score',fontsize=fontsize)
plt.ylabel('Second Harmonic Score',fontsize=fontsize)
plt.savefig('../figures/OneClassSVM.png', bbox_inches='tight')

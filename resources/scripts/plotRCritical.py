import numpy as np
from icecube import MLSandboxPythonAccess
import sys
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy import stats
def plot_rcritical(file_name,cl):
    import pylab
    import matplotlib.pyplot as plt
    import pickle
    ranks = MLSandboxPythonAccess.FCRanks()
    ranks.load(file_name)

    xis = np.linspace(0,0.06,300)
    jet = cm = plt.get_cmap('jet')
    values = range(len(cl))
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    for i in range(len(cl)):
        ranks.SetConfidenceLevel(float(cl[i]))
        critB = list()
        for xi in xis:
            critB.append(ranks.rCB(xi))
        colorVal = scalarMap.to_rgba(values[i])

        plt.plot(xis,np.array(critB),color=colorVal,label='CL %f '%float(cl[i]))
        plt.axhline(-stats.chi2.ppf(float(cl[i]),1)/2,color=colorVal,ls = 'dashed')
    plt.ylabel("ln(R)")
    plt.xlabel("likelihood parameter")
    plt.legend()
    plt.show()

if (__name__ == "__main__"):
    file_name = sys.argv[1]
    cl = sys.argv[2:]
    #if(len(cl))
    plot_rcritical(file_name,cl)

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

def setup_llh(sig, bg, x,  N, seed):
    sig_exp = MLSandboxPythonAccess.Distribution(sig, min(x), max(x), 1)
    bg_exp = MLSandboxPythonAccess.Distribution(bg, min(x), max(x), 1)
    llh = MLSandboxPythonAccess.Likelihood.BinnedLikelihood.ShapeLikelihood(sig_exp, bg_exp, sig_exp, bg_exp, N, seed)
    return llh

def load_expectations():
    '''Test data provided by mzoll (solar wimp pdfs)
    '''
    import pickle
    fd = open('../data/data_pdf.pkl','r')
    fs = open('../data/wimp_pdf.pkl','r')
    sig = pickle.load(fs)
    bg = pickle.load(fd)
    x = np.linspace(1,180,len(sig))-0.5
    return (sig,bg,x)

def compute_sensitivity():
    import pylab
    import matplotlib.pyplot as plt

    #or we load some
    (sig,bg,x) = load_expectations()
    #Plotting the pdfs for fun ;)


    #Here we set the number of events in our fake analysis using a
    #binned shape likelihood
    N = 7060
    llh = setup_llh(sig, bg, x,  N, 1)
    #Since it's a binned likelihood we can histogram the events to
    #speed up llh evaluations
    llh.EnableHistogramedEvents()
    llh.EnablePoissonSampling()
    #setting up the analysis
    analysis = MLSandboxPythonAccess.FeldmanCousinsAnalysis(llh = llh,
                                                            cl = 0.9, #confidence level (not really important yet)
                                                           )
    plt.show()
    #Compute rank distributions for different llh parameter xi as set up above
    #n_threads key argument sets number of threads to use during this step
    #analysis.ComputeRanks( n_experiments = 10000, #Number of pseudo experiments (trials)
    #                       min_xi = 0.0, #lower boundary of likelihood parameter for rank calculation
    #                       max_xi = 100.0/N, #upper boundary likelihood parameter for rank calculation
    #                       n_steps = 100, #number of steps
    #                       n_threads = 6)

    #since computing the ranks distributions takes a lot of CPU time/resources
    #analysis.ranks.save("ranks_example.dat")
    ranksobj = MLSandboxPythonAccess.FCRanks()
    ranksobj.load("ranks_example2.dat")
    #Pybindings are still missing for the function which makes an ensemble of
    #pseudo experiments and computes limits,
    #To determine the median upper limit when assuming bg only
    #we do it explicitly in python for now.
    up_lim = list()
    down_lim = list()
    analysis.SetFCRanks(ranksobj)
    analysis.ranks.SetConfidenceLevel(0.9)
    analysis.Sample(0)
    #Compute FC limits for this particular experiments
    (up, down) = analysis.ComputeLimits()

    import dashi
    dashi.visual()
    ranks = ranksobj.get_ranks()
    bedges = (np.linspace(0,200.0,100),np.linspace(-1,10,100))
    rankhist = dashi.histogram.hist2d(binedges = bedges)
    for k in ranks.keys():
        rankhist.fill((np.ones(len(ranks[k]))*float(k)*N,ranks[k] ))

    xi = np.linspace(0,200.0/N,100)
    ts = []
    critB = []
    for x in xi:
        ts.append(analysis.EvaluateTestsStatistic(x))
        critB.append(analysis.ranks.rCB(x))
    print(ts)
    print(critB)
    plt.figure()
    rankhist.imshow(log=True)
    plt.plot(xi*N,critB,'k')
    #plt.plot(xi*N,ts,'k')
    analysis.Sample(20.0/N)
    ts = []
    for x in xi:
        ts.append(analysis.EvaluateTestsStatistic(x))
    #plt.plot(xi,ts,'k')
    plt.colorbar()
    plt.show()

compute_sensitivity()

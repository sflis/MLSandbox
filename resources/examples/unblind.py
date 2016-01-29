import numpy as np
from icecube import MLSandboxPythonAccess

#Here we create some binned expectations (PDFs) for our binned likelihood
def create_expectations():

    from scipy import stats
    bins= 1000
    x = np.linspace(0,10,bins)
    #Our background is an exponential
    bg = np.ones(bins)*stats.expon.pdf(x,scale = 5)
    #Our signal is a gaussian
    sig = stats.norm.pdf(x,loc = 4,scale=0.51)
    return (sig,bg,x)



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

    from scipy import stats
    #We can generating some pdfs
    #(sig,bg,x) = create_expectations()
    #or we load some
    (sig,bg,x) = load_expectations()

    #Plotting the pdfs for fun ;)
    plt.plot(x, sig, '*r')
    plt.plot(x, bg, 'ob')

    #Here we set the number of events in our fake analysis using a
    #binned shape likelihood
    #N = 9354
    N = 35319
    llh = setup_llh(sig,#signal expectation (numpy array)
                    bg, #background expectation (numpy array)
                    x,  #This can be an array or just a tuple wich describes the range of your pdfs
                    N,  #Number of events in the sample
                    10  #Seed to rng
                    )

    #Since it's a binned likelihood we can histogram the events to
    #speed up llh evaluations
    llh.EnableHistogramedEvents()
    llh.EnablePoissonSampling()

    #setting up the analysis
    analysis = MLSandboxPythonAccess.FeldmanCousinsAnalysis(llh = llh,
                                                            cl = 0.9, #confidence level (not really important yet)
                                                        )

    ranks = MLSandboxPythonAccess.FCRanks()
    ranks.load("ranks_example.dat")
    #since computing the ranks distributions takes a lot of CPU time/resources
    analysis.SetFCRanks(ranks)

    #Pybindings are still missing for the function which makes an ensemble of
    #pseudo experiments and computes limits,
    #To determine the median upper limit when assuming bg only
    #we do it explicitly in python for now.
    up_lim = list()
    down_lim = list()
    analysis.ranks.SetConfidenceLevel(0.9)
    for i in range(10000):
        #Generate a pseudo experiment with background only
        analysis.Sample(0)
        #Compute FC limits for this particular experiments
        (up, down) = analysis.ComputeLimits()
        #save them..
        up_lim.append(up)
        down_lim.append(down)

    up_lim = sorted(up_lim)
    up_lim = np.array(up_lim)
    median_lim = np.median(up_lim)
    print("Median upper limit: %f"%(median_lim*N))


    #Normaly the unblinded sample would be loaded here and set with:
    #unblinded_events = load_unblinded_events() #numpy array
    #llh.SetEventSample(unblinded_events)
    #But since we don't have any we just make a fake sample
    analysis.Sample(0.0/N)
    #Compute the limits of the unblinded sample
    (up, down) = analysis.ComputeLimits()
    #Extract the best fit.
    bestfit = analysis.minimizer.bestFit

    from icecube.MLSandboxPythonAccess import utils

    p_value,p_value_sigma = utils.pval(up,up_lim)
    print("Unblinded best fit: %f events"%bestfit)
    print("Unblinded upper limit: %f"%(up*N))
    print("p-value: %f"%p_value)
    print("sigma: %f"%p_value_sigma)
    sigma_bands = utils.sigma_bands_central(up_lim,[1,2,3,4])
    for k,v in sigma_bands.iteritems():
        print ("%d sigma band (%f): (%f,%f)"%(k,v[0],v[1]*N,v[2]*N))
    plt.figure()
    hist, bins = np.histogram(np.array(up_lim), 50, (0, 0.05))
    bincent = bins[:-1] + np.diff(bins)/2.0
    plt.plot(bincent*N, hist, 'ro')

    plt.axvline(median_lim*N,color = 'black',linestyle = 'solid')

    hist, bins = np.histogram(np.array(down_lim), 50, (0, 0.05))
    bincent = bins[:-1] + np.diff(bins)/2.0
    plt.plot(bincent*N, hist, 'bo')

    plt.show()
if (__name__ == "__main__"):
    compute_sensitivity()

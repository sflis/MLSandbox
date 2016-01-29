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
    fd = open("../data/data_WINTER_IC_Up_020_pdf_U0.pkl",'r')
    fs = open("../data/wimp_m50ch11_WINTER_DC_Up_020_pdf_U0.pkl",'r')
    fed = open("../data/EnergyPDF_data_WINTER_DC_Up.pkl",'r')
    fes = open("../data/EnergyPDF_wimp_m50ch11_WINTER_DC_Up.pkl",'r')

    sig = pickle.load(fs)
    bg = pickle.load(fd)
    sige = pickle.load(fes)
    bge = pickle.load(fed)
    itemindex = np.where(bge==0)
    print(itemindex)
    bge = bge[:itemindex[0][0]]
    sige = sige[:itemindex[0][0]]
    sig = np.outer(sig,sige).flatten()
    bg = np.outer(bg,bge).flatten()
    print(len(sig))
    print(len(bg))
    x = np.linspace(1,180,len(sig))-0.5
    return (sig,bg,x)

def compute_sensitivity():
    import pylab
    import matplotlib.pyplot as plt

    #We can generating some pdfs
    #(sig,bg,x) = create_expectations()
    #or we load some
    (sig,bg,x) = load_expectations()
    #Plotting the pdfs for fun ;)

#    plt.plot(x,sig,'*r')
#    plt.plot(x,bg,'ob')

    #Here we set the number of events in our fake analysis using a
    #binned shape likelihood
    #N = 9354#1
    N = 6126
    #N += np.sqrt(N)
    #N= 5400
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
    computeRanks = False
    if(computeRanks):
        #Compute rank distributions for different llh parameter xi as set up above
        #n_threads key argument sets number of threads to use during this step
        analysis.ComputeRanks( n_experiments = 10000, #Number of pseudo experiments (trials)
                               min_xi = 0.0, #lower boundary of likelihood parameter for rank calculation
                               max_xi = 100.0/N, #upper boundary likelihood parameter for rank calculation
                               n_steps = 20, #number of steps
                               n_threads = 7)

        #since computing the ranks distributions takes a lot of CPU time/resources
        analysis.ranks.save("ranks_example.dat")
    else:
        ranksobj = MLSandboxPythonAccess.FCRanks()
        ranksobj.load("ranks_example.dat")
        analysis.SetFCRanks(ranksobj)
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
    #up_lim = sorted(up_lim)
    median_lim = np.median(np.array(up_lim))
    #N -= np.sqrt(N)
    print("Median upper limit: %f"%(median_lim*N))
    plt.figure()
    hist, bins = np.histogram(np.array(up_lim),50,(0,0.05))
    bincent = bins[:-1]+np.diff(bins)/2.0
    plt.plot(bincent*N,hist,'ro')

    plt.axvline(median_lim*N,color = 'black',linestyle = 'solid')

    hist, bins = np.histogram(np.array(down_lim),50,(0,0.05))
    bincent = bins[:-1]+np.diff(bins)/2.0
    plt.plot(bincent*N,hist,'bo')
    plt.show()

if (__name__ == "__main__"):
    compute_sensitivity()

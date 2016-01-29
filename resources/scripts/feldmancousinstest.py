import numpy as np
from icecube import MLSandboxPythonAccess

def create_expectations():
    from scipy import stats
    bins= 1000
    x = np.linspace(0,10,bins)
    bg = np.ones(bins)*stats.expon.pdf(x,scale = 5)
    sig = stats.norm.pdf(x,loc = 4,scale=0.51)
    return (sig,bg,x)


def setup_llh(sig, bg, x,  N, seed):
    sig = [s for s in sig]
    bg = [b for b in bg]
    sig_exp = MLSandboxPythonAccess.Distribution(sig, min(x), max(x), 1)
    bg_exp = MLSandboxPythonAccess.Distribution(bg, min(x), max(x), 1)
    llh = MLSandboxPythonAccess.Likelihood.BinnedLikelihood.ShapeLikelihood(sig_exp, 
                                                                            bg_exp, 
                                                                            sig_exp, 
                                                                            bg_exp, N, seed)
    return llh
def run_test():
    import pylab
    import matplotlib.pyplot as plt

    (sig,bg,x) = create_expectations()
    
    plt.plot(x,sig,'*r')
    plt.plot(x,bg,'ob')
    N = 16000
    llh = setup_llh(sig, bg, x,  N, 2)
    llh.EnableHistogramedEvents()
    
    analysis = MLSandboxPythonAccess.FeldmanCousinsAnalysis(llh, 0.5)
    analysis.ComputeRanks(1000,0, .10, 3,1)
    
    
    #up_lim = list()
    #down_lim = list()
    #n_s = 0
    ##print("Injecting %d signal events"%n_s)
    #mu = np.linspace(0,0.11,1000)
    
    #critB = list()
    #analysis.GetRanks.SetConfidenceLevel(0.9)
    #for m in mu:
        #critB.append(analysis.GetRanks.rCB(m))
        ##print(analysis.GetRanks.rCB(m))
    #plt.figure()
    #plt.plot(mu*N,np.array(critB))
    
    #critB = list()
    #analysis.GetRanks.SetConfidenceLevel(0.8)
    #for m in mu:
        #critB.append(analysis.GetRanks.rCB(m))
    #plt.plot(mu*N,np.array(critB),'r')
    
    #critB = list()
    #analysis.GetRanks.SetConfidenceLevel(0.7)
    #for m in mu:
        #critB.append(analysis.GetRanks.rCB(m))
    #plt.plot(mu*N,np.array(critB),'g')
    ##for i in range(10000):
    #analysis.GetRanks.SetConfidenceLevel(0.9)
    ##analysis.GetRanks.save("test_ranks.dat")
    
    #analysis.Sample(n_s)
    #(up,down) = analysis.ComputeLimits()
    #teststatistic = list()
    #for m in mu:
        #teststatistic.append(analysis.EvaluateTestsStatistic(m))
        
        
        
    #plt.plot(mu*N,np.array(teststatistic),'k')
    #plt.axvline(up*N,color='k', linestyle='solid')
    #plt.ylim(-2,0)
    #print(up*N)
    
    
    #analysis.Sample(80.0/N)
    #(up,down) = analysis.ComputeLimits()
    #teststatistic = list()
    #for m in mu:
        #teststatistic.append(analysis.EvaluateTestsStatistic(m))
        
        

    #plt.plot(mu*N,np.array(teststatistic),'k')
    #plt.axvline(up*N,color='k', linestyle='solid')
    #plt.ylim(-2,0)
    #print(up*N,down*N)
    
    #up_lim = list()
    #down_lim = list()
    #for i in range(10000):
        #analysis.Sample(0)
        #(up,down) = analysis.ComputeLimits()
        #up_lim.append(up)
        #down_lim.append(down)
    
    #plt.figure()
    #hist, bins = np.histogram(np.array(up_lim),50,(0,0.05))
    #bincent = bins[:-1]+np.diff(bins)/2.0
    #plt.plot(bincent*N,hist,'ro')
    
    #hist, bins = np.histogram(np.array(down_lim),50,(0,0.05))
    #bincent = bins[:-1]+np.diff(bins)/2.0
    #plt.plot(bincent*N,hist,'bo')
 
    #plt.show()
if (__name__ == "__main__"):
    run_test()
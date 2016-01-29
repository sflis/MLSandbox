import numpy as np
from icecube import MLSandboxPythonAccess

def create_expectations():
    from scipy import stats
    bins= 180
    x = np.linspace(0,10,bins)
    bg = np.ones(bins)*stats.expon.pdf(x,scale = 5)
    sig = stats.norm.pdf(x,loc = 7,scale=.1)
    return (sig,bg,x)

def load_exp():
    import pickle
    fd = open('../examples/data_pdf.pkl','r')
    fs = open('../examples/wimp_pdf.pkl','r')


    sig = pickle.load(fs)
    bg = pickle.load(fd)
    x = np.linspace(1,180,len(sig))-0.5
    return (sig,bg,x)
def run_test():
    import pylab
    import matplotlib.pyplot as plt


    (sig,bg,x) = create_expectations()
    #(sig,bg,x) = load_exp()
    #Plotting the pdfs for fun ;)
    plt.plot(x,sig,'*r')
    plt.plot(x,bg,'ob')

    mini = MLSandboxPythonAccess.Minimizer()
    sig_exp = MLSandboxPythonAccess.Distribution(sig, min(x), max(x), 1)
    bg_exp = MLSandboxPythonAccess.Distribution(bg, min(x), max(x), 1)
    N = 9354.0
    llh = MLSandboxPythonAccess.Likelihood.BinnedLikelihood.ShapeLikelihood(sig_exp, bg_exp, sig_exp, bg_exp, N,3)
    #llh = MLSandboxPythonAccess.Likelihood.BinnedLikelihood.SignalContaminatedLH(sig_exp, bg_exp, sig_exp, bg_exp, N,3)
    best_fits = list()
    llh.EnableHistogramedEvents()
    llh.EnablePoissonSampling()
    n_s = 0.0

    best_median  = list()
    nss = np.linspace(0,99,100)
    for ns in nss:
        best_fits = list()
        iterations = list()
        print("================================")
        print("Injecting %f signal events"%ns)
        for i in range(10000):
            llh.SampleEvents(ns/N)
            mini.ComputeBestFit(llh)
            #print("best fit: %f  llh value: %f"%(mini.bestFit*100, mini.BestFitLLH))
            best_fits.append(mini.bestFit)
            iterations.append(mini.nIterations)
        best_median.append(np.median(np.array(best_fits)))
        print("median best fit: %f"%(np.median(np.array(best_fits))*N))
        print("average best fit: %f"%(np.mean(np.array(best_fits))*N))
        print("deviation: %f"%((ns-(np.median(np.array(best_fits))*N))/ns*100))
        print("average iterations: %f"%np.mean(np.array(iterations)))
    plt.figure()
    plt.plot(nss[1:],np.array(best_median[1:])*N/nss[1:])
    plt.show()
if (__name__ == "__main__"):
    run_test()

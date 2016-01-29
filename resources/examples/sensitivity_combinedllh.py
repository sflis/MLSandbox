#!/usr/bin/env python

import pickle, math
import numpy as np
from icecube import MLSandboxPythonAccess
from optparse import OptionParser
from scipy import stats, integrate


def setup_llh(sig, bg, x,  N, seed):
    print(len(sig), min(x)-.5, max(x)+.5)
    sig_exp = MLSandboxPythonAccess.Distribution(sig, min(x), max(x), 1)
    bg_exp = MLSandboxPythonAccess.Distribution(bg, min(x), max(x), 1)
    sig_exp.SetCDFSampling(True)
    bg_exp.SetCDFSampling(True)
    llh = MLSandboxPythonAccess.Likelihood.BinnedLikelihood.ShapeLikelihood(sig_exp, bg_exp, sig_exp, bg_exp, N, seed)
    return llh

def load_expectations(sig_file, bg_file):
    '''Test data provided by mzoll (solar wimp pdfs)
    '''
    import pickle
    fs = open(sig_file,'r')
    fd = open(bg_file,'r')
    sig = pickle.load(fs)
    bg = pickle.load(fd)
    x = np.linspace(0,179,len(sig))+0.5
    return (sig,bg,x)

def compute_sensitivity(options):
    import pylab
    import matplotlib.pyplot as plt
    data_pdf1 = "../data/data_WINTER_IC_Up_020_pdf_U0.pkl"
    sig_pdf1 = "../data/wimp_m500ch5_WINTER_IC_Up_020_pdf_U0.pkl"
    #sig_pdf1 = "../data/wimp_m250ch11_WINTER_IC_Up_020_pdf_U0.pkl"
    data_pdf2 = "../data/data_WINTER_DC_Up_020_pdf_U0.pkl"
    sig_pdf2 = "../data/wimp_m500ch5_WINTER_DC_Up_020_pdf_U0.pkl"
    #sig_pdf2 = "../data/wimp_m250ch11_WINTER_DC_Up_020_pdf_U0.pkl"
    weights = [0.8,0.2]

    s = 0.0079+0.0055
    weights = [0.0079/s,0.0055/s]
    print(weights)
    #weights = [0.5,0.5]
    #or we load some
    (sig1,bg1,x1) = load_expectations(sig_pdf1, data_pdf1)
    (sig2,bg2,x2) = load_expectations(sig_pdf1, data_pdf1)


    N1 = 21055
    N2 = 6126
    N = N1+N2
    llh1 = setup_llh(sig1, bg1, x1,  N1, 1)
    llh2 = setup_llh(sig2, bg2, x2,  N2, 2)
    #Since it's a binned likelihood we can histogram the events to
    #speed up llh evaluations
    llh1.EnableHistogramedEvents()
    llh1.EnablePoissonSampling()

    llh2.EnableHistogramedEvents()
    llh2.EnablePoissonSampling()
    llhc = MLSandboxPythonAccess.Likelihood.CombinedLikelihood([llh1,llh2],weights)

    #setting up the analysis
    analysis = MLSandboxPythonAccess.FeldmanCousinsAnalysis(llh = llhc,
                                                            cl = 0.9, #confidence level (not really important yet)
                                                           )

    if (options.RANKS != None): #there are precomputed values available
        ranks = MLSandboxPythonAccess.FCRanks()
        ranks.load(options.RANKS)
        analysis.SetFCRanks(ranks)
    else: #everything needs to be recomputed
        analysis.ComputeRanks( n_experiments = options.NEXP, #Number of pseudo experiments (trials)
                           min_xi = options.MUMIN/N, #lower boundary of likelihood parameter for rank calculation
                           max_xi = options.MUMAX/N, #upper boundary likelihood parameter for rank calculation
                           n_steps = int(options.NSTEPS), #number of steps
                           n_threads = int(options.NTHREADS))
        analysis.ranks.save(options.OUTPUT+"ranks.dat")

    #Pybindings are still missing for the function which makes an ensemble of
    #pseudo experiments and computes limits,
    #To determine the median upper limit when assuming bg only
    #we do it explicitly in python for now.
    up_lim = list()
    down_lim = list()
    analysis.ranks.SetConfidenceLevel(0.9)
    for i in range(options.NEXP):
        #Generate a pseudo experiment with background only
        analysis.Sample(0)
        #Compute FC limits for this particular experiments
        (up, down) = analysis.ComputeLimits()
        #save them..
        up_lim.append(up)
        down_lim.append(down)


    print "=== Sensitivity ==="
    up_lim = sorted(up_lim)
    up_lim = np.array(up_lim)
    median_lim = np.median(up_lim)
    print("Median upper limit: %f"%(median_lim*N))
    for j in xrange(5):
        val, err = integrate.quad(stats.norm.pdf, -(j+1),j+1)
        print ("upwards %d sigma (%f): %f"%(j+1, val, np.percentile(up_lim, (val+(1.-val)/2)*100)*N))
    for j in xrange(5):
        val, err = integrate.quad(stats.norm.pdf, -(j+1),j+1)
        print ("downwards %d sigma (%f): %f"%(j+1, val, np.percentile(up_lim, ((1.-val)/2)*100)*N))



if (__name__ == "__main__"):
  parser = OptionParser('usage: %prog [options]')
  parser.add_option("-o", "--output", action="store", type="string", default='', dest="OUTPUT", help="destination for output")
  parser.add_option("-e", "--events", action="store", type="string", default=None, dest="EVENTS", help="pickled array of event-values")
  parser.add_option("-s", "--signal", action="store", type="string", default=None, dest="SIGNAL", help="pickled Signal PDF as NARRAY")
  parser.add_option("-b", "--background", action="store", type="string", default=None, dest="BG", help="pickled Background PDF as NARRAY")
  parser.add_option("-r", "--ranks", action="store", type="string", default=None, dest="RANKS", help="stored Ranks;load the file otherwise will be recomputed")
  #parser.add_option("--nevents", action="store", type="int", default=None, dest="NEVENTS", help="number of events")
  parser.add_option("--min_mu", action="store", type="float", default=0., dest="MUMIN", help="minimum number of inj signal events")
  parser.add_option("--max_mu", action="store", type="float", default=0., dest="MUMAX", help="maximum number of inj signal events")
  parser.add_option("--nexperiments", action="store", type="int", default=10000, dest="NEXP", help="number of pseudo-experiments")
  parser.add_option("--nsteps", action="store", type="int", default=None, dest="NSTEPS", help="number of steps between MINMU and MAXMU")
  parser.add_option("-j", action="store", type="int", default=1, dest="NTHREADS", help="use that many threads")
  (options,args) = parser.parse_args()

  if (options.NSTEPS==None):
    options.NSTEPS = options.MUMAX


  compute_sensitivity(options)

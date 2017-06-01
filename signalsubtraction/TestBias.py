import numpy as np
import MLSandbox

import matplotlib.pyplot as plt
import pickle, math
import numpy as np
import MLSandbox
from optparse import OptionParser
from scipy import stats, integrate
import dashi
dashi.visual()

def setup_llh():

    def setup_llh(sig, bg,sig_samp, x,  N, seed):
        sig_exp = MLSandbox.Distribution(sig, min(x), max(x), 1)
        bg_exp = MLSandbox.Distribution(bg, min(x), max(x), 1)
        sig_samp_exp = MLSandbox.Distribution(sig_samp, min(x), max(x), 1)
        sig_exp.SetCDFSampling(True)
        bg_exp.SetCDFSampling(True)
        sig_samp_exp.SetCDFSampling(True)
        llh = MLSandbox.Likelihood.BinnedLikelihood.ShapeLikelihood(sig_exp, bg_exp,sig_samp_exp, bg_exp, N, seed)
        llh.EnablePoissonSampling()
        return llh
def plotpdfs(pdfs):
    plt.plot(pdfs['sig'])
    plt.plot(pdfs['bg'])
    plt.plot(pdfs['sig_scr'])
    plt.figure()
    plt.plot(pdfs['data_binned'])
    plt.show()

def setup_sc_llh_p(pdfs, seed=1):
        x = [1,10]
        sig_exp = MLSandbox.Distribution(pdfs['sig'], min(x), max(x), 1)
        bg_exp = MLSandbox.Distribution(pdfs['bg'], min(x), max(x), 1)
        sig_samp_exp = MLSandbox.Distribution(pdfs['sig'], min(x), max(x), 1)
        sig_exp_scr = MLSandbox.Distribution(pdfs['sig_scr'], min(x), max(x), 1)
        sig_samp_exp_scr = MLSandbox.Distribution(pdfs['sig_scr'], min(x), max(x), 1)
        sig_samp_exp_scr0 = MLSandbox.Distribution(np.zeros(pdfs['sig_scr'].shape), min(x), max(x), 1)
        sig_exp.SetCDFSampling(True)
        bg_exp.SetCDFSampling(True)
        sig_samp_exp.SetCDFSampling(True)
        sig_exp_scr.SetCDFSampling(True)
        sig_samp_exp_scr.SetCDFSampling(True)
        #plotpdfs(pdfs)

        model = MLSandbox.Likelihood.BinnedLikelihood.PSignalContaminatedLH.Model.Binomial
        llh = MLSandbox.Likelihood.BinnedLikelihood.PSignalContaminatedLH(sig_exp,
                                                                        bg_exp,
                                                                        sig_exp_scr,
                                                                        sig_samp_exp,
                                                                        bg_exp,
                                                                        sig_samp_exp_scr,   
                                                                        pdfs['n_data'],
                                                                        pdfs['sig_fract'],
                                                                        pdfs['bg_fract'],
                                                                        model,
                                                                        pdfs['sig_fract'],
                                                                        pdfs['bg_fract'],
                                                                        seed)
        return llh    
def setup_sc_llh(pdfs, seed=1):
        x = [1,10]
        sig_exp = MLSandbox.Distribution(pdfs['sig'], min(x), max(x), 1)
        bg_exp = MLSandbox.Distribution(pdfs['bg'], min(x), max(x), 1)
        sig_samp_exp = MLSandbox.Distribution(pdfs['sig'], min(x), max(x), 1)
        sig_exp_scr = MLSandbox.Distribution(pdfs['sig_scr'], min(x), max(x), 1)
        sig_samp_exp_scr = MLSandbox.Distribution(pdfs['sig_scr'], min(x), max(x), 1)
        sig_samp_exp_scr0 = MLSandbox.Distribution(np.zeros(pdfs['sig_scr'].shape), min(x), max(x), 1)
        sig_exp.SetCDFSampling(True)
        bg_exp.SetCDFSampling(True)
        sig_samp_exp.SetCDFSampling(True)
        sig_exp_scr.SetCDFSampling(True)
        sig_samp_exp_scr.SetCDFSampling(True)
        #plotpdfs(pdfs)

        model = MLSandbox.Likelihood.BinnedLikelihood.SignalContaminatedLH.Model.Binomial
        if(pdfs['bg_fract']==1):
            model = MLSandbox.Likelihood.BinnedLikelihood.SignalContaminatedLH.Model.None
        llh = MLSandbox.Likelihood.BinnedLikelihood.SignalContaminatedLH(sig_exp,
                                                                        bg_exp,
                                                                        sig_exp_scr,
                                                                        sig_samp_exp,
                                                                        bg_exp,
                                                                        sig_samp_exp_scr,   
                                                                        pdfs['n_data'],
                                                                        pdfs['sig_fract'],
                                                                        pdfs['bg_fract'],
                                                                        model,
                                                                        pdfs['sig_fract'],
                                                                        pdfs['bg_fract'],
                                                                        seed)
        model = MLSandbox.Likelihood.BinnedLikelihood.LikelihoodCollection.Model.Binomial
        if(pdfs['bg_fract']==1):
            model = MLSandbox.Likelihood.BinnedLikelihood.LikelihoodCollection.Model.None
        llh_plain = MLSandbox.Likelihood.BinnedLikelihood.LikelihoodCollection(sig_exp,
                                                                        bg_exp,
                                                                        sig_exp_scr,
                                                                        sig_samp_exp,
                                                                        bg_exp,
                                                                        sig_samp_exp_scr,      
                                                                        pdfs['n_data'],
                                                                        pdfs['sig_fract'],
                                                                        pdfs['bg_fract'],
                                                                        model,
                                                                        pdfs['sig_fract'],
                                                                        pdfs['bg_fract'],
                                                                        seed)

        llh_shape = MLSandbox.Likelihood.BinnedLikelihood.ShapeLikelihood(sig_exp, 
                                                                            bg_exp,
                                                                            sig_samp_exp, 
                                                                            bg_exp, 
                                                                            float(pdfs['n_data']*pdfs['bg_fract']), 
                                                                            seed)
        llh_shape.EnablePoissonSampling()
        #print(sum(pdfs['data_binned']))
        #print(pdfs['sig_fract'],pdfs['bg_fract'])
        return {#'correct_correction':llh,
                'collection':llh_plain,
                #'plain_shape':llh_shape
                }


def run_test(ns,n_trials,selection,bias_col):

    best_fits = list()
    for k,v in bias_col.iteritems():
        if(ns in v.keys()):
            continue
        else:
            v[ns] = list()

    mini = MLSandbox.Minimizer()
    for i in range(n_trials):
        pdfs = selection.generate_pdfs(ns, decl=30, ra=60)
        llhs = setup_sc_llh(pdfs)
        for k,llh in llhs.iteritems():
            if(k == 'collection'):
                #continue
                for l in ['standardSigSub','noSigSubCorr','NonTerminatedSigSub','HybridSigSub']:
                    llh.SetLLHFunction(l)
                    llh.SetEvents(pdfs['data_binned'])
                    mini.ComputeBestFit(llh)
                    #print("best fit: %f  "%(mini.bestFit*pdfs['n_data']))
                    bias_col[l][ns].append(mini.bestFit*pdfs['n_data'])
            else:
                llh.SetEvents(pdfs['data_binned'])
                mini.ComputeBestFit(llh)
                #print("best fit: %f  "%(mini.bestFit*pdfs['n_data']))
                bias_col[k][ns].append(mini.bestFit*pdfs['n_data'])

def run_test_p(ns,n_trials,selection,bias_col,w2xis ):

    best_fits = list()
    for k,v in bias_col.iteritems():
        if(ns in v.keys()):
            continue
        else:
            v[ns] = list()

    mini = MLSandbox.Minimizer()
    for i in range(n_trials):
        pdfs = selection.generate_pdfs(ns, decl=30, ra=60)
        llh = setup_sc_llh_p(pdfs)
        for k in w2xis:#llhs.iteritems():
            llh.SetEvents(pdfs['data_binned'])
            llh.SetW2Xi(k)
            mini.ComputeBestFit(llh)
            #print("best fit: %f  "%(mini.bestFit*pdfs['n_data']))
            bias_col[k][ns].append(mini.bestFit*pdfs['n_data'])



if (__name__ == "__main__"):
    import ToySimulation as toy_sim
    import sys
    import time
    seed = int(sys.argv[1])
    #reco = sys.argv[2]
    w2xis = np.linspace(0,1,20)
    bias = dict()
    #for w2xi in w2xis:
    #    bias[w2xi] = dict()
    #bias['plain_shape'] = dict()
    #bias['original_correction'] = dict()
    #bias['correct_correction'] = dict()
    bias['standardSigSub'] = dict()
    bias['noSigSubCorr'] = dict()
    bias['NonTerminatedSigSub'] = dict()
    bias['HybridSigSub'] = dict()
    #bias['no_correction'] = dict()
    def d(x):
        return x
    bg = toy_sim.bg_model(2.5e4, d, seed=seed)
    sig = toy_sim.sig_model(20*np.pi/180, seed=seed)
    selection = toy_sim.ToySimulation(bg, sig, ra_fraction = 1./1.)
    n_trials = 100
    time0 = time.time()
    count = 1

    import pickle
    for i in range(60):
        for ns in np.linspace(0,200,21):
            try:
                time1 = time.time()
                run_test(int(ns), n_trials, selection, bias)#,w2xis)
                time2 = time.time()
                dtime = time2-time1
                print(dtime)
                print('Current time per trial %f s'%((time2-time1)/n_trials))
                print('Average time per trial %f s'%((time2-time0)/(n_trials*count)))
                print('Trials %d done out of %d'%(count*n_trials,n_trials*60*41))
                count += 1
            except Exception, e:
                 print('error',e)
            with open('data/bias_col_window_20deg_%d.pkl'%seed, 'w') as f:
                    pickle.dump(bias,f)
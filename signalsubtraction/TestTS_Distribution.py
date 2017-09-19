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

def plotpdfs(pdfs):
    plt.plot(pdfs['signal_pdf'])
    plt.plot(pdfs['data_scr_pdf'])
    plt.plot(pdfs['signal_scr_pdf'])
    plt.figure()
    plt.plot(pdfs['binned_scr_data'])
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
        #plotpdfs(pdfs)
        x = [1,10]
        sig_exp = MLSandbox.Distribution(pdfs['signal_pdf'], min(x), max(x), 1)
        bg_exp = MLSandbox.Distribution(pdfs['data_scr_pdf'], min(x), max(x), 1)
        sig_samp_exp = MLSandbox.Distribution(pdfs['signal_pdf'], min(x), max(x), 1)
        sig_exp_scr = MLSandbox.Distribution(pdfs['signal_scr_pdf'], min(x), max(x), 1)
        sig_samp_exp_scr = MLSandbox.Distribution(pdfs['signal_scr_pdf'], min(x), max(x), 1)
        sig_samp_exp_scr0 = MLSandbox.Distribution(np.zeros(pdfs['signal_scr_pdf'].shape), min(x), max(x), 1)
        sig_exp.SetCDFSampling(True)
        bg_exp.SetCDFSampling(True)
        sig_samp_exp.SetCDFSampling(True)
        sig_exp_scr.SetCDFSampling(True)
        sig_samp_exp_scr.SetCDFSampling(True)


        model = MLSandbox.Likelihood.BinnedLikelihood.SignalContaminatedLH.Model.Binomial
        if(pdfs['bg_fract']==1.0):
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
        if(pdfs['bg_fract']==1.0):
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
        return {#'correct_correction':llh,
                'collection':llh_plain,
                #'plain_shape':llh_shape
                }
import healpy



def compute_TS_distr(dec ,n_trials,selection,bias_col):

    best_fits = list()
    for k,v in bias_col.iteritems():
        if(not isinstance(v,dict) or (dec in v.keys())):
            continue
        else:
            v[dec] = list()

    mini = MLSandbox.Minimizer()
    selection.generate_data_sample(0)

    for i in range(n_trials):
        pdfs = selection.generate_pdfs()
        llhs = setup_sc_llh(pdfs)
        if(i%100==0):
            print(i)
        for k,llh in llhs.iteritems():
            if(k == 'collection'):
                for l in ['standardSigSub','noSigSubCorr','HybridSigSub']:
                    try:
                        llh.SetLLHFunction(l)
                        llh.SetEvents(pdfs['binned_scr_data'])
                        mini.ComputeBestFit(llh)
                        bias_col[l][dec].append(mini.bestFitLLH-llh.EvaluateLLH(0.0))
                        #print(mini.bestFit)
                    except Exception, e:
                        print('Error with %s likelihood:' %l,e)


            else:
                llh.SetEvents(pdfs['binned_scr_data'])
                mini.ComputeBestFit(llh)
                bias_col[k][dec].append(mini.bestFitLLH-llh.EvaluateLLH(0.0))


if (__name__ == "__main__"):
    import ToySimulationHealpix as toy_sim
    #import SimpleToyModel as toy_sim
    import sys
    import time
    import copy
    seed = int(sys.argv[1])
    source_ext = float(sys.argv[2])
    n_events = float(sys.argv[3])
    source_model = sys.argv[4]
    #reco = sys.argv[2]
    w2xis = np.linspace(0,1,20)
    bias = dict()
    bias['standardSigSub'] = dict()
    bias['noSigSubCorr'] = dict()
    bias['NonTerminatedSigSub'] = dict()
    bias['HybridSigSub'] = dict()
    ts_distr = copy.deepcopy(bias)
    #bias['no_correction'] = dict()

    bg_shape='flat'#='linear_slope'#'fisher60'
    #n_events = 2.5e5
    upper_sig_frac = n_events*0.08
    #source_ext = 5
    n_side=64
    bg = toy_sim.bg_model(n_events, toy_sim.bg_models[bg_shape], seed=seed)
    data = dict()
    if(source_model == 'SingleSource'):
        dec = -50
        sig = toy_sim.sig_model(source_ext*np.pi/180,(dec*np.pi/180,266*np.pi/180), n_side = n_side, seed=seed)
        data['source_ext'] = source_ext
        data['source_model'] = 'SingleSource'
    elif(source_model == 'ComplexSource'):
        #dec = -29
        sig = toy_sim.complexsource(2.*np.pi/180, nsources=int(source_ext), n_side = n_side, seed=seed)
        data['n_sources'] = source_ext
        data['source_ext'] = 2.
        data['source_model'] = 'ComplexSource'
        #data['declination'] = dec

    #sig = toy_sim.sig_model(source_ext*np.pi/180,(-80*np.pi/180,266*np.pi/180), n_side = n_side, seed=seed)
    
    selection = toy_sim.ToySimulation(bg, sig, n_side = n_side)


    n_trials = 10000
    time0 = time.time()
    count = 1
    
    data['nevents'] = n_events
    data['source_ext'] = source_ext
    data['bg_shape'] = bg_shape
    data['n_side'] = n_side

    #pdfs = selection.generate_pdfs(int(n_events*0.9), decl=n_dec_bins, ra=30)
    #plotpdfs(pdfs)

    import pickle
    for i in range(1):
        for dec in np.array([85.,60.,30.,0.])*np.pi/180.:#np.linspace(-np.pi/2*0.95,np.pi/2*0.95,100):
            print(dec*180/np.pi)
            try:
                time1 = time.time()
                if(source_model == 'SingleSource'):
                    sig = toy_sim.sig_model(source_ext*np.pi/180,(dec*np.pi/180,266*np.pi/180), n_side = n_side, seed=seed)
                elif(source_model == 'ComplexSource'):
                    sig = toy_sim.complexsource(2.*np.pi/180, nsources=int(source_ext), n_side = n_side, seed=seed)
                #sig = toy_sim.sig_model(source_ext*np.pi/180,(dec,266*np.pi/180), n_side = n_side, seed=seed)
                selection = toy_sim.ToySimulation(bg, sig, n_side = n_side)
                compute_TS_distr(dec, n_trials, selection, ts_distr)

                time2 = time.time()
                dtime = time2-time1
                print(dtime)
                print('Current time per trial %f s'%((time2-time1)/n_trials))
                print('Average time per trial %f s'%((time2-time0)/(n_trials*count)))
                #print('Trials %d done out of %d'%(count*n_trials,n_trials*60*41))
                #print('Trials %f  done'%(count*n_trials/float(n_trials*60*41)))

                count += 1
            except Exception, e:
                print('error',e)
            data['bias_data'] = bias
            data['ts_distr'] = ts_distr
            with open('data/ts_distr_nevents_%g_SourceExt_%g_BgShape_%s_n_side_%d_seed%d.pkl'%(n_events,source_ext,bg_shape,n_side,seed), 'w') as f:
                   pickle.dump(data,f)
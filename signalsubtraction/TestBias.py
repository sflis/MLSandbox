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

def run_test(ns,n_trials,selection,bias_col):

    best_fits = list()
    for k,v in bias_col.iteritems():
        if(not isinstance(v,dict) or (ns in v.keys())):
            continue
        else:
            v[ns] = list()

    mini = MLSandbox.Minimizer()
    mini.SetBoundaries(-1.,1.0)
    for i in range(n_trials):
        selection.generate_data_sample(ns)
        pdfs = selection.generate_pdfs()
        llhs = setup_sc_llh(pdfs)

        for k,llh in llhs.iteritems():
            if(k == 'collection'):
                #continue
                for l in ['standardSigSub','noSigSubCorr','NonTerminatedSigSub']:
                    try:

                        llh.SetLLHFunction(l)
                        llh.SetEvents(pdfs['binned_data'])
                        mini.ComputeBestFit(llh)
                        bias_col[l][ns].append(mini.bestFit*pdfs['n_data'])


                    except Exception, e:
                        xs = np.linspace(-0.0002,0.015,600)
                        y = list()
                        for x in xs:
                            y.append(np.log(-llh.EvaluateLLH(x)))
                        plt.figure()
                        plt.plot(xs,y, color='black')
                        plt.axvline(mini.bestFit, color='red')
                        print(mini.bestFit)
                        print(xs[np.argmin(y)])
                        plt.axvline(xs[np.argmin(y)], color='blue',ls='--',lw=3)
                        plt.show()

                        print('Error with %s likelihood:' %l,e)
            else:
                continue
                llh.SetEvents(pdfs['binned_data'])
                mini.ComputeBestFit(llh)
                bias_col[k][ns].append(mini.bestFit*pdfs['n_data'])

def compute_TS_distr(selection):    
    mini = MLSandbox.Minimizer()
    selection.generate_data_sample(0)
    pdfs = selection.generate_pdfs()
    llh = setup_sc_llh(pdfs)['collection']
    xs = np.linspace(-0.0002,0.015,600)

    try:
        llh.SetLLHFunction('noSigSubCorr')
        llh.SetEvents(pdfs['binned_scr_data'])
        #print(np.sum(np.isnan(pdfs['binned_scr_data'])))
        #print(np.sum((pdfs['data_scr_pdf'])<=0))
        mini.ComputeBestFit(llh)

    except Exception, e:
        print('Error with %s likelihood:' %'standardSigSub',e)
                        
    y = list()
    for x in xs:
        y.append(np.log(-llh.EvaluateLLH(x)))
    plt.figure()
    plt.plot(xs,y, color='black')
    plt.axvline(mini.bestFit, color='red')
    print(mini.bestFit)
    print(xs[np.argmin(y)])
    plt.axvline(xs[np.argmin(y)], color='blue',ls='--',lw=3)
    plt.show()


if (__name__ == "__main__"):
    import ToySimulationHealpix as toy_sim
    import sys
    import time
    import copy
    seed = int(sys.argv[1])
    source_model = sys.argv[2]
    ext_arg = float(sys.argv[3])
    n_events = float(sys.argv[4])
    dec = float(sys.argv[5])
    #reco = sys.argv[2]
    w2xis = np.linspace(0,1,20)
    bias = dict()
    bias['standardSigSub'] = dict()
    bias['noSigSubCorr'] = dict()
    bias['NonTerminatedSigSub'] = dict()
    bias['HybridSigSub'] = dict()
    ts_distr = copy.deepcopy(bias)


    bg_shape='flat'#='linear_slope'#'fisher60'
    #n_events = 2.5e5
    upper_sig_frac = n_events*0.02
    

    n_side=64
    bg = toy_sim.bg_model(n_events, toy_sim.bg_models[bg_shape], seed=seed)
    data = dict()
    data['nevents'] = n_events
    data['source_model'] = source_model
    data['bg_shape'] = bg_shape
    data['n_side'] = n_side

    if(source_model == 'SingleSource'):
        #dec = -50
        sig = toy_sim.sig_model(ext_arg*np.pi/180,(dec*np.pi/180,266*np.pi/180), n_side = n_side, seed=seed)
        data['source_ext'] = ext_arg
        data['declination'] = dec
    elif(source_model == 'ComplexSource'):
        dec = -29
        sig = toy_sim.complexsource(2.*np.pi/180, nsources=int(ext_arg), n_side = n_side, seed=seed)
        data['n_sources'] = ext_arg
        data['source_ext'] = 2.
        data['declination'] = dec
    
    selection = toy_sim.ToySimulation(bg, sig, n_side = n_side)
    n_trials = 10
    time0 = time.time()
    count = 1


    #pdfs = selection.generate_pdfs(int(n_events*0.9), decl=n_dec_bins, ra=30)
    #plotpdfs(pdfs)

    import pickle
    for i in range(100):
        for j,ns in enumerate(np.linspace(0,upper_sig_frac,30)):
            print(ns)
            try:
                time1 = time.time()
                run_test(int(ns), n_trials+int(80./(j**2+1)), selection, bias)#,w2xis)compute_TS_distr(selection)#
                time2 = time.time()
                dtime = time2-time1
                print(dtime)
                print('Current time per trial %f s'%((time2-time1)/n_trials))
                print('Average time per trial %f s'%((time2-time0)/(n_trials*count)))
                print('Trials %d done out of %d'%(count*n_trials,n_trials*60*41))
                print('Trials %f  done'%(count*n_trials/float(n_trials*60*41)))

                count += 1
            except Exception, e:
                print('error',e)
            data['bias_data'] = bias
            data['ts_distr'] = ts_distr

            with open('data/bias_nevents_%g_SourceModel_%s_%g_dec_%g_BgShape_%s_n_side_%d_seed%d.pkl'%(n_events,source_model,ext_arg,dec,bg_shape,n_side,seed), 'w') as f:
                   pickle.dump(data,f)
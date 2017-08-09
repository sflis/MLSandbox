import matplotlib.pyplot as plt
import glob
import pickle
import dashi
import numpy as np
from utils import  *
dashi.visual()







def create_page(data):
    label_names = {'standardSigSub':'Standard Signal Subtraction',
               'NonTerminatedSigSub':'Non Terminated Subtraction',
               'noSigSubCorr':'No signal subtraction',
               'HybridSigSub':'Hybrid Subtraction'}


    path = "/home/sflis/i3/projects/signalsubtraction/page/source/"


    bias = data['bias_data']
    tag = "SourceExt%dBgShape%sNEvents%dDecBins%d"%(data['source_ext'],data['bg_shape'],data['nevents'],data['n_dec_bins'])
    f = open(path+"SourceExt%dBgShape%sNEvents%dDecBins%d.rst"%(data['source_ext'],data['bg_shape'],data['nevents'],data['n_dec_bins']),'w')
    title = "SourceExt, %d BgShape %s, NEvents %d, DecBins %d,"%(data['source_ext'],data['bg_shape'],data['nevents'],data['n_dec_bins'])
    #s = "===\n"
    #s += title+'\n'
    #s + = "===\n"
    #s + = header
    with open("Template.rst",'r') as f2:
        template = f2.read(-1)

    template = make_title(title,'title',template)

    publisher = PlotTablePublisher(title)

    import matplotlib as mpl
    s = 0.8
    mpl.rcParams['xtick.labelsize'] = 18
    mpl.rcParams['ytick.labelsize'] = 18
    x = np.linspace(0.000,0.08,100)
    fig1 = plt.figure(figsize=(15*s,10*s))
    plt.plot(x,x,color='black',lw=3)
    fig2 = plt.figure(figsize=(15*s,10*s))
    plt.plot(x,x,color='black',lw=3)
    fig3 = plt.figure(figsize=(15*s,10*s))

    import matplotlib.colors as colors
    import matplotlib.cm as cm
    jet = colors.Colormap('jet')
    #cNorm  = colors.Normalize(vmin=0, vmax=1)
    #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    #cNorm  = colors.Normalize(vmin=0, vmax=0)
    #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    i  = 0.
    ls = ['--',':','-.','--']
    nevents = data['nevents']
    for k in bias.keys():
        if(k=='plain_shape'):
            continue
        if(isinstance(k,float)):
            cv = cm.viridis(k)#scalarMap.to_rgba(float(k))
        else:
            cv = cm.Set1(i/len(bias.keys()))
            
        #if('no' in k):
        #    continue
        av = list()
        median = list()
        ns = list()
        av_err = list()
        median_err = list()
        for n in bias[k].keys():
            av.append(np.mean(bias[k][n]))
            av_err.append(np.std(bias[k][n]))
            ns.append(n)
            median.append(np.median(bias[k][n]))
        ns, av,median,av_err = zip(*sorted(zip(ns,av,median,av_err)))
        ns = np.array(ns)/nevents
        av = np.array(av)/nevents
        median = np.array(median)/nevents
        av_err = np.array(av_err)/nevents
        print(av_err)

        #fig1 = plt.figure()
        plt.figure(fig1.number)
        plt.plot(ns,av,label=label_names[k],color=cv,ls=ls[int(i)],lw=3)#yerr=av_err
        #fig2 = plt.figure()
        plt.figure(fig2.number)
        plt.errorbar(ns,median,yerr=av_err,label=label_names[k],color=cv)
        #fig3 = plt.figure()
        plt.figure(fig3.number)
        plt.errorbar(ns[1:],av[1:]/(ns[1:]),yerr=av_err[1:]/ns[1:],label=label_names[k],ls='None',marker='o',color=cv,)
        if(k==0 or k==1):
            plt.figure(fig2.number)
            plt.errorbar(ns, av, yerr=av_err,color='red',lw=2)
            plt.figure(fig3.number)
            plt.errorbar(ns,median,yerr=av_err,color='red',lw=2)
        i+=1.0
    fontsize = 35        
    plt.figure(fig3.number)

    plt.hlines(1,min(ns[1:]),max(ns[1:]))
    #plt.legend(loc='best',frameon=False)
    #plt.xlabel('True signal fraction',size=26)
    #plt.ylabel('(Average best fit)/(number of injected signal events)')
    #plt.ylim((0.8,1.2))
    #plt.ylim((0.95,1.05))
    #plt.xlim((0,np.max(ns/2.5e2)))

    plt.figure(fig1.number)#,figsize=(15,10))
    plt.tick_params(labelsize=25)
    plt.xlabel('True signal fraction',size=fontsize)
    plt.ylabel('Average of best fit',size=fontsize)

    #plt.ylim((0.0001,0.008))
    #plt.yscale('log')
    plt.legend(loc='best',frameon=False,fontsize=28)
    plt.tight_layout()
    plt.savefig(path+"_images/"+tag+"AverageFitvsTrue.png")
    publisher.add_figures("_images/"+tag+"AverageFitvsTrue.png","AverageFitvsTrue")

    #plt.savefig(path+'SignalSubtractionBias.pdf')

    plt.figure(fig2.number)
    plt.legend(loc='best',frameon=False)
    plt.xlabel('Number of injected signal events')
    plt.ylabel('Median of best fit')
    plt.tight_layout()
    plt.savefig(path+"_images/"+tag+"MedianFitvsTrue.png")
    publisher.add_figures("_images/"+tag+"MedianFitvsTrue.png","AverageFitvsTrue")
    plt.figure(fig3.number)
    plt.ylabel('Relative bias')
    plt.xlabel('Number of injected signal events')
    plt.tight_layout()
    plt.savefig(path+"_images/"+tag+"RelativeBiasMedian.png")
    publisher.add_figures("_images/"+tag+"RelativeBiasMedian.png","AverageFitvsTrue")
    #t = [('Description','Plot')]

    #t.append((path+"_images/"+tag+"AverageFitvsTrue.png","AverageFitvsTrue"))
    #t.append((path+"_images/"+tag+"MedianFitvsTrue.png","AverageFitvsTrue"))
    #t.append((path+"_images/"+tag+"RelativeBiasMedian.png","AverageFitvsTrue"))

    #S = as_rest_table(t)
    S = publisher.publish()
    template = insert(S,'plot_table',template)
    sim_table = [('Description','value')]
    sim_table.append(('Source extension',':math:`%d^\circ`'%data['source_ext']))
    sim_table.append(('Background model','%s'%data['bg_shape']))
    sim_table.append(('Number of events','%d'%data['nevents']))
    sim_table.append(('Number of declination bins','%s'%data['n_dec_bins']))
    str_stim_table = as_rest_table(sim_table)
    template = insert(str_stim_table,'sim_table',template)
    


    from ToySimulation import *
    seed = 1
    d =bg_models[data['bg_shape']]
    bg = bg_model(data['nevents'],d,decl_band=30*np.pi/180,seed =seed)
    sig = sig_model(data['source_ext']*np.pi/180,decl_band=30*np.pi/180,seed = seed)
    selection = ToySimulation(bg,sig,ra_fraction = 1./1.,decl_band=30*np.pi/180)
    pdfs = selection.generate_pdfs(0,decl = data['n_dec_bins'],ra=120)
    plt.figure()
    pdfs['sig_hist'].imshow()
    plt.title('signal')
    plt.xlabel("RA a.u")
    plt.ylabel("DEC a.u")
    plt.colorbar()
    plt.savefig(path+"_images/"+tag+"Signal2d.png")
    template = insert('.. image:: %s \n    :scale: 40'%("_images/"+tag+"Signal2d.png") + ' %\n',
        "signal_2d",
        template)

    plt.figure()
    pdfs['bg_hist'].imshow()
    plt.title('background')
    plt.xlabel("RA a.u")
    plt.ylabel("DEC a.u")
    plt.colorbar()
    plt.savefig(path+"_images/"+tag+"Background2d.png")
    template = insert('.. image:: %s \n    :scale: 40'%("_images/"+tag+"Background2d.png") + ' %\n ',
        "background_2d",
        template)


    plt.figure()
    pdfs['bg_decl'].line(label='data')
    pdfs['bg_decl_pdf'].line(label='smooth pdf')
    plt.title('data')
    plt.xlabel("DEC a.u")
    plt.ylabel("Events a.u")
    plt.legend()
    plt.savefig(path+"_images/"+tag+"BackgroundDec.png")
    template = insert('.. image:: %s \n    :scale: 45'%("_images/"+tag+"BackgroundDec.png") + ' %\n ',
        "background_dec",
        template)

    plt.figure()
    pdfs['data_binned_hist'].imshow()
    plt.title('data')
    plt.xlabel("RA a.u")
    plt.ylabel("DEC a.u")
    plt.savefig(path+"_images/"+tag+"Data.png")
    template = insert('.. image:: %s \n    :scale: 45'%("_images/"+tag+"Data.png") + ' %\n ',
        "data_bg_only",
        template)    

    bg = bg_model(data['nevents'],d,decl_band=30*np.pi/180,seed =seed)
    sig = sig_model(data['source_ext']*np.pi/180,decl_band=30*np.pi/180,seed = seed)
    selection = ToySimulation(bg,sig,ra_fraction = 1./1.,decl_band=30*np.pi/180)
    pdfs = selection.generate_pdfs(int(0.5*data['nevents']),decl  = data['n_dec_bins'],ra=120)


    plt.figure()
    pdfs['bg_decl'].line(label='data')
    pdfs['bg_decl_pdf'].line(label='smooth pdf')
    plt.title('data')
    plt.xlabel("DEC a.u")
    plt.ylabel("Events a.u")
    plt.legend()
    plt.savefig(path+"_images/"+tag+"BackgroundDec0.5.png")
    template = insert('.. image:: %s \n    :scale: 45'%("_images/"+tag+"BackgroundDec0.5.png") + ' %\n ',
        "background_dec_0.5",
        template)

    plt.figure()
    pdfs['data_binned_hist'].imshow()
    plt.title('data')
    plt.xlabel("RA a.u")
    plt.ylabel("DEC a.u")
    plt.savefig(path+"_images/"+tag+"Data0.5.png")
    template = insert('.. image:: %s \n    :scale: 45'%("_images/"+tag+"Data0.5.png") + ' %\n ',
        "data_sig_0.5",
        template)

    plt.figure()
    pdfs['bg_decl_pdf'].line(label='scr. data')
    sigscr = pdfs['sig_scr_hist'].project([1])
    sigscr.line(label='scr. signal')
    for w in [0.1,0.3,0.4,0.5]:
        plt.step(sigscr.bincenters,pdfs['bg_decl_pdf'].bincontent-w*sigscr.bincontent,where='mid',label='background est. %g'%w)
    plt.title('data')
    plt.xlabel("RA a.u")
    plt.ylabel("Events a.u")
    plt.legend(ncol=2,frameon=False)
    plt.savefig(path+"_images/"+tag+"ScrDataVsSignal.png")
    template = insert('.. image:: %s \n    :scale: 45'%("_images/"+tag+"ScrDataVsSignal.png") + ' %\n ',
        "scrdatasig",
        template)

    plt.figure()
    pdfs['bg_decl'].line(label='scr. data')
    sigscr = pdfs['sig_scr_hist'].project([1])
    sigscr.line(label='scr. signal')
    for w in [0.1,0.3,0.4,0.5]:
        plt.step(sigscr.bincenters,pdfs['bg_decl'].bincontent-w*sigscr.bincontent,where='mid',label='background est. %g'%w)
    plt.title('data')
    plt.xlabel("RA a.u")
    plt.ylabel("Events a.u")
    plt.legend(ncol=2,frameon=False)
    plt.savefig(path+"_images/"+tag+"ScrDataVsSignal2.png")
    template = insert('.. image:: %s \n    :scale: 45'%("_images/"+tag+"ScrDataVsSignal2.png") + ' %\n ',
        "scrdatasig2",
        template)




    f.write(template)
    f.close()
    return tag

if __name__ == '__main__':
    files = glob.glob("data/*.pkl")#bias_nevents_2500_SourceExt_5_BgShape_linear_slope_n_dec_bins_20_seed1.pkl")
    #data = pickle.load(open(files[0],'r'))


    contents = list()
    for f in files:
        
        try:
            data = pickle.load(open(f,'r'))
            contents.append(create_page(data))
        except Exception, e:   
            print(e)
            continue
    print(contents)
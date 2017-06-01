import numpy as np
import MLSandbox
from MLSandbox import Distribution

'''
class Distribution(object):
    def __init__(self, distr_array,minv,maxv,seed=1):
        self.distr_array = distr_array/np.sum(distr_array)
        self.cumul_distr_array = np.cumsum(self.distr_array)
        self.bin_size = (maxv-minv)/self.distr_array.size
        self.seed = seed
        self.minv = minv
        self.maxv = maxv
        self.bins = np.arange(self.distr_array.size)
    def Sample(self):
        #r = np.random.uniform(0.,1.)
        return self.minv + np.argmax(self.cumul_distr_array<np.random.uniform(0.,1.))*self.bin_size + self.bin_size/2*np.random.uniform(0.,1.)
        #return self.minv + np.random.choice(self.distr_array,self.bins)*self.bin_size + self.bin_size/2*np.random.uniform(0.,1.)
    def SampleInteger(self):
        return np.argmax(self.cumul_distr_array>np.random.uniform(0.,1.))
'''


def vectorize_distr(d):
    '''Helper function that returns a vectorize version of the Sample method of a Distribution instance'''
    def vectorized(n):
        r = np.empty(n)
        for i in xrange(n):
            r[i] = d.Sample()
        return r
    return vectorized



class bg_model(object):
    def __init__(self, mean, decl_dep, decl_band=30*np.pi/180, seed=1):
        '''This class describes the background for the toy MC
            
            args:
            mean -- the mean of the total number of background events
            decl_dep -- A reference to a function that describes the declination dependence of the background.
            decl_band -- The band in declination (rad) which the background is defined.
            seef -- Seed to the random number generator
        '''
        self.mean = mean
        self.decl_dep = decl_dep
        x = np.linspace(0,1,1000)
        y = decl_dep(x)
        ddec = Distribution(y,-decl_band/2,decl_band/2,seed)
        self.z = ddec.SampleN#vectorize_distr(ddec)
        self.decl_band = decl_band
    def generate(self, n = None):
        '''Generates a background sample

            args:
            n -- number of background events in the sample if None n 
                 is drawn from a poisson distribution with mean self.mean (Default: None)

            Returns:
            A list with two numpy array with the event coordinates in [right ascension, declination]
        '''
        if(n == None):
            nbg = np.random.poisson(self.mean)
        else:
            nbg = n
        return [np.random.uniform(-1,1,nbg)*np.pi, self.z(nbg) ]




class sig_model(object):
    def __init__(self,psf,decl_band=30*np.pi/180,seed= 1):
        '''This class describes a Gaussian signal for the toy MC
            
            args:
            psf -- the point spread function of the signal used as the sigma value of the Gaussian 
            decl_band -- The band in declination (rad) which the background is defined.
            seef -- Seed to the random number generator

        '''
        self.psf = psf
        self.seed = seed
        from scipy import stats
        
        x = np.linspace(-decl_band/2,decl_band/2,2000)
        y = stats.norm.pdf(x,scale=self.psf)
        ddec = Distribution(y,min(x),max(x),self.seed)
        x = np.linspace(-np.pi,np.pi,2000)
        y = stats.norm.pdf(x,scale=self.psf)
        dra = Distribution(y,min(x),max(x),self.seed+1)

        self.ddecv = ddec.SampleN#vectorize_distr(ddec)
        self.drav = dra.SampleN#vectorize_distr(dra)

    def generate(self,nsig):
        '''Generates signal events

            args:
            nsig -- The number of signal events to be generated

            Returns:
            A list with two numpy array with the event coordinates in [right ascension, declination]
        '''
        return [self.drav(nsig),self.ddecv(nsig)]

    def pdf(self,decl,ra):
        ''' Return a binned signal pdf

            args:
            decl -- The bin centers in declination
            ra -- The bin centers in right ascension
        '''
        from scipy import stats
        return stats.norm.pdf(decl,scale=self.psf) * stats.norm.pdf(ra,scale=self.psf)



class ToySimulation(object):
    """ This class performs a simulation given a background and signal model



    """
    def __init__(self,bg_model,sig_model,ra_fraction=1/6.,decl_band=30*np.pi/180):
        """ 
            args:
            bg_model -- The background model 
            sig_model -- The signal model
            ra_fraction -- How much of the right ascension should be used in the shape analysis
            decl_band -- The size of the declination band in ra
        """
        self.bg_model = bg_model
        self.sig_model = sig_model
        self.data = np.empty(0)
        self.decl_band = decl_band
        self.ra_fraction = ra_fraction
        self.generated_sig_pdf = False

    def generate_data_sample(self,nsig):
        '''Generates a data sample with nsig injected 
        signal events
        '''
        N = self.bg_model.mean
        self.data = self.bg_model.generate(int(N-nsig))
        tmp_data = self.sig_model.generate(nsig)
        
        self.data =[np.append(self.data[0],tmp_data[0]),np.append(self.data[1],tmp_data[1])]
        return self.data

    def generate_pdfs(self,nsig,decl=60,ra=120):
        """ Performs a pseudo experiment with nsig number of signal events.

            The background pdf is determined for each pseudo experiment an depends on the 
            sampled data

            args:
            nsig -- Number of signal events
            decl -- Number of bins in declination
            ra -- Number of bins in right ascension

            Returns:
            A dictionary with pdfs and pseudo data
        """
        import dashi
        
        ra = np.linspace(-self.ra_fraction*np.pi,self.ra_fraction*np.pi,ra)
        decl = np.linspace(-self.decl_band/2,self.decl_band/2,decl)
        pdfs = dict()
        
        pdfs['data_raw'] = self.generate_data_sample(nsig)
        pdfs['bg_fract'] = self.ra_fraction

        #oversampling signal to determine pdf and signal fraction
        #Only once!
        if(self.generated_sig_pdf == False):
            self.generated_sig_pdf = True
            N = int(8e6)
            tmp = self.sig_model.generate(N)
            self.sig_fract = np.sum(np.abs(tmp[0])<self.ra_fraction*np.pi)/float(N)
            #pdfs['sig_fract'] = self.sig_fract

            #generating signal pdf
            self.sig_pdf = dashi.histogram.hist2d((ra,decl))
            self.sig_pdf.fill((tmp[0],tmp[1]))
            self.sig_pdf.bincontent[:] /= np.sum(self.sig_pdf.bincontent)


            #scramble the signal
            self.sigdec =  self.sig_pdf.project([1])
            self.sig_pdf_scr = dashi.histogram.hist2d((ra,decl))
            for i in range(self.sig_pdf.bincontent.shape[1]):
                self.sig_pdf_scr.bincontent[:,i] = self.sigdec.bincontent[i]
            self.sig_pdf_scr.bincontent[:] /= np.sum(self.sig_pdf_scr.bincontent)
        pdfs['sig_scr'] = self.sig_pdf_scr.bincontent.flatten([1])

        pdfs['sig_fract'] = self.sig_fract
        #generating background pdf from data which is scrambled in RA
        bg_pdf = dashi.histogram.hist2d((ra,decl))
        bg_pdf.fill((np.random.uniform(-1,1,len(pdfs['data_raw'][0]))*np.pi*2,pdfs['data_raw'][1]))
        bgdec = bg_pdf.project([1])
        for i in range(bg_pdf.bincontent.shape[1]):
            bg_pdf.bincontent[:,i] = bgdec.bincontent[i]
        bg_pdf.bincontent /= np.sum(bg_pdf.bincontent)

        #bin the data
        data = dashi.histogram.hist2d((ra,decl))
        data.fill((pdfs['data_raw'][0],pdfs['data_raw'][1]))

        
        pdfs['data_binned'] = np.array(data.bincontent.flatten([1]),dtype=np.uint64)
        pdfs['sig'] = self.sig_pdf.bincontent.flatten([1])
        pdfs['bg'] = bg_pdf.bincontent.flatten([1])
        pdfs['data_binned_hist'] = data
        pdfs['sig_hist'] = self.sig_pdf
        pdfs['sig_scr_hist'] = self.sig_pdf_scr
        pdfs['bg_hist'] = bg_pdf
        pdfs['n_data'] = len(pdfs['data_raw'][0])
        return pdfs


if (__name__ == "__main__"):
    import dashi
    import matplotlib.pyplot as plt
    dashi.visual()
    def d(x):
        return x
    bg = bg_model(2.5e5,d)
    sig = sig_model(2*np.pi/180)
    selection = ToySimulation(bg,sig)
    pdfs = selection.generate_pdfs(50,decl = 30,ra=60)
    plt.figure()
    pdfs['sig_hist'].imshow()
    plt.title('signal')
    plt.xlabel("RA a.u")
    plt.ylabel("DEC a.u")
    plt.colorbar()

    plt.figure()
    pdfs['sig_scr_hist'].imshow()
    plt.title('signal scrambled')
    plt.xlabel("RA a.u")
    plt.ylabel("DEC a.u")

    plt.colorbar()

    plt.figure()
    pdfs['bg_hist'].imshow()
    plt.colorbar()

    plt.title('scrambled data')
    plt.xlabel("RA a.u")
    plt.ylabel("DEC a.u")

    plt.figure()
    pdfs['data_binned_hist'].imshow()
    plt.title('data')
    plt.xlabel("RA a.u")
    plt.ylabel("DEC a.u")

    plt.colorbar()
    print(pdfs['sig_fract'])
    print(pdfs['bg_fract'])
    plt.show()

import numpy as np
import MLSandbox
from MLSandbox import Distribution
import healpy
import dashi

class bg_model(object):
    def __init__(self, mean, decl_dep,  seed=1):
        '''This class describes the background for the toy MC
            
            args:
            mean -- the mean of the total number of background events
            decl_dep -- A reference to a function that describes 
                        the declination dependence of the background in the theta angle, i.e [0,pi] 
            seef -- Seed to the random number generator
        '''
        self.mean = mean
        self.decl_dep = decl_dep
        x = np.linspace(0,np.pi,1000)
        y = decl_dep(x)
        ddec = Distribution(y,0,np.pi,seed)
        self.ddec = ddec
        self.z = ddec.SampleN#vectorize_distr(ddec)
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

        return [np.random.uniform(0,2,nbg)*np.pi, self.z(nbg) ]




class sig_model(object):
    def __init__(self,psf,coord,n_side,seed= 1):
        '''This class describes a Gaussian signal for the toy MC
            
            args:
            psf -- the point spread function of the signal used as the sigma value of the Gaussian 
            coord -- a tuple or a list that defines the position of the source in the sky (ra,dec)
            n_side -- the n-side of the  healpix map
            seed -- Seed to the random number generator

        '''
        self.psf = psf
        self.seed = seed        
        #A 'true' signal pdf in as a healpix with small bins
        self.n_side = n_side*2
        n_pix = healpy.nside2npix(self.n_side)
        self.true_pdf = np.zeros(n_pix)

        inds = np.arange(0,n_pix)
        angs = healpy.pix2ang(self.n_side,inds)
        # for ga_dec,ga_ra in zip(self.ga_sources_dec,self.ga_sources_ra):
        self.true_pdf[:] +=fisher(angs[0],angs[1],np.pi/2-coord[0],coord[1],1./psf**2)

        # for i in xrange(n_pix):
        #     pix_pos = healpy.pix2ang(self.n_side,i)
        #     self.true_pdf[i] =  fisher(pix_pos[0],pix_pos[1],np.pi/2-coord[0],coord[1],1./psf**2)
        
        self.distr = Distribution(self.true_pdf,0,1,self.seed)
        
        #creating the signal pdf that will be used in the likelihood   
        self.binned_pdf = healpy.ud_grade(self.true_pdf, self.n_side/2)    
        

    def generate(self,nsig):
        '''Generates signal events

            args:
            nsig -- The number of signal events to be generated

            Returns:
            A list with two numpy array with the event coordinates in [right ascension, declination]
        '''
        pixs = self.distr.SampleIN(nsig)
        pos  = healpy.pix2ang(self.n_side,pixs)
        return [pos[1],pos[0]]

    def pdf(self):
        ''' Return a binned signal pdf

            args:
            decl -- The bin centers in declination
            ra -- The bin centers in right ascension
        '''
        return self.binned_pdf

class complexsource(object):
    def __init__(self,psf,n_side,nsources,seed= 1):
        '''This class describes a complex source composed out of several Gaussian signal sources in the galactic plane
            
            args:
            psf -- the point spread function of the signal used as the sigma value of the Gaussians 
            n_side -- the n-side of the  healpix map
            nsources -- the number of individual sources 
            seed -- Seed to the random number generator

        '''
        self.psf = psf
        self.seed = seed        
        #A 'true' signal pdf in as a healpix with small bins
        self.n_side = n_side*2
        n_pix = healpy.nside2npix(self.n_side)
        self.true_pdf = np.zeros(n_pix)
        
        #Sample source locations in the galactic plane
        self.ga_sources_ra = np.random.normal(loc=0,scale=0.5,size=nsources)*np.pi#np.random.uniform(0,2,nsources)*np.pi
        self.ga_sources_dec = np.random.normal(loc=0,scale=0.061,size=nsources)*np.pi
        #Rotate the coordinates to Equatorial 
        r = healpy.Rotator(coord=['G','C'])
        self.ga_sources_dec, self.ga_sources_ra = r(np.pi/2-self.ga_sources_dec, self.ga_sources_ra)  # Apply the conversion
        
        #Sum up contributions from all sources in the healpix map 
        inds = np.arange(0,n_pix)
        angs = healpy.pix2ang(self.n_side,inds)
        for ga_dec,ga_ra in zip(self.ga_sources_dec,self.ga_sources_ra):
            self.true_pdf[:] +=fisher(angs[0],angs[1],ga_dec,ga_ra,1./psf**2)
        #Creating a binned distribution from the healpix map that we can sample from    
        self.distr = Distribution(self.true_pdf,0,1,self.seed)
        
        #creating the signal pdf that will be used in the likelihood   
        self.binned_pdf = healpy.ud_grade(self.true_pdf, self.n_side/2)
        
    
        
        
        

    def generate(self,nsig):
        '''Generates signal events

            args:
            nsig -- The number of signal events to be generated

            Returns:
            A list with two numpy array with the event coordinates in [right ascension, declination]
        '''
        pixs = self.distr.SampleIN(nsig)
        pos  = healpy.pix2ang(self.n_side,pixs)
        return [pos[1],pos[0]]

    def pdf(self):
        ''' Return a binned signal pdf

            args:
            decl -- The bin centers in declination
            ra -- The bin centers in right ascension
        '''
        return self.binned_pdf



def fisher(x1,x2,p1,p2,k):
    '''Fisher distribution numerically stable for 1/k**2>0.04
    '''
    if(k<200):
        return k/(4*np.pi*np.sinh(k))*np.exp(k*(
                                                np.sin(x1)*np.sin(p1)*np.cos(x2-p2)
                                                + np.cos(x1)*np.cos(p1)
                                                ) )
    else:
        return k/(2*np.pi)*np.exp(k*(
                                                np.sin(x1)*np.sin(p1)*np.cos(x2-p2)
                                                + np.cos(x1)*np.cos(p1)
                                                -1) )
 

def projected_fisher(ang,theta,thetap):
    ''' A fisher distribution projected on to the theta axis

        args:
        ang -- The angular uncertainty (sqrt(var))
        theta -- The theta angle at which the function should be evaluated
        thetap -- The center in theta of the fisher distribution
    '''
    from scipy import special
    k=1/ang**2
    e = np.exp(k*np.cos(theta)*np.cos(thetap))
    bessel = special.i0(k*np.sin(theta)*np.sin(thetap))*np.sin(theta)
    return k*e*bessel/(2*np.sinh(k))


def gettheta(n_side):
    all_ang = healpy.pix2ang(n_side,np.arange(healpy.nside2npix(n_side)))
    thetas = set(all_ang[0])
    tdiff = np.diff(np.asarray(sorted(list(thetas))))
    bedges =np.empty(len(thetas)+1)
    bedges[1:-1] = np.array(sorted(list(thetas)))[:-1]+tdiff/2
    bedges[0] = 0
    bedges[-1] = np.pi
    return bedges, all_ang 


class ToySimulation(object):
    """ This class performs a simulation given a background and signal model



    """
    def __init__(self, bg_model, sig_model, n_side = 16):
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
        self.generated_sig_pdf = False
        self.n_side = n_side
        self.n_pix = healpy.nside2npix(self.n_side)
        bedges, all_ang = gettheta(self.n_side)
        self.bedges = bedges
        #generate scrambled signal pdf
        self.determine_sig_pdf()
    
    def compute_scr_data_pdf(self):
        ''' Determines the 'background' pdf from the scrambled data
        '''
        #Get the declination distribution of the data
        zenith_pdf =  dashi.histogram.hist1d(self.bedges)
        zenith_pdf.fill(np.array(self.data[1]))
        
        solid_ang = 2*np.pi*(np.cos(self.bedges[:-1])-np.cos(self.bedges[1:]))
        zenith_pdf.bincontent[:] *=healpy.nside2pixarea(self.n_side)/solid_ang
        
        #Map the declination distribution back on the healpix map
        self.scr_data_pdf = np.zeros(self.n_pix)
        for i in xrange(len(zenith_pdf.binedges[:-1])):
            inds = healpy.query_strip(self.n_side,zenith_pdf.binedges[i],zenith_pdf.binedges[i+1])
            angs = healpy.pix2ang(self.n_side,inds)
            self.scr_data_pdf[inds] = zenith_pdf.bincontent[i]
     

    def generate_data_sample(self,nsig):
        ''' Generates a data sample with nsig injected 
            signal events
        '''
        N = self.bg_model.mean
        self.data = self.bg_model.generate(int(N-nsig))
        tmp_data = self.sig_model.generate(nsig)
        
        self.data =[np.append(self.data[0], tmp_data[0]), np.append(self.data[1],tmp_data[1])]
        self.compute_scr_data_pdf()
        self.newdata = True
        return self.data
    
    def determine_sig_pdf(self):
        ''' Determines the normal and scrambled signal pdf 
        '''
        sky_pdf = self.sig_model.pdf()
        zenith_pdf =  dashi.histogram.hist1d(self.bedges)
        self.sky_pdf_scr = np.zeros(self.n_pix)
        #Integrating the signal in RA
        for i in xrange(len(zenith_pdf.binedges[:-1])):
            inds = healpy.query_strip(self.n_side,zenith_pdf.binedges[i],zenith_pdf.binedges[i+1])
            angs = healpy.pix2ang(self.n_side,inds)
            solid_ang = 2*np.pi*(np.cos(zenith_pdf.binedges[i])-np.cos(zenith_pdf.binedges[i+1]))
            zenith_pdf.fill(np.array(angs[0]),np.array(sky_pdf[inds]/solid_ang))# np.array([angs[0]),np.array(sky_pdf[inds]/solid_ang))
            self.sky_pdf_scr[inds] = zenith_pdf.bincontent[i]*healpy.nside2pixarea(self.n_side)
     


    def generate_pdfs(self):
        """ Determines the pdfs for the current sample and also generates a scrambled dataset

            Returns:
            A dictionary with pdfs and pseudo data
        """
        import dashi

        n_pix = healpy.nside2npix(self.n_side)

        pdfs = dict()
        pdfs['data_raw'] = self.data
        pdfs['n_data'] = len(pdfs['data_raw'][0])
        pdfs['bg_fract'] = 1
        pdfs['sig_fract'] = 1

        

        pdfs['signal_pdf'] = self.sig_model.pdf()
        pdfs['signal_scr_pdf'] = self.sky_pdf_scr
        pdfs['data_scr_pdf'] = self.scr_data_pdf

        if(self.newdata):#Only do this step if a new data set is generated
          
            pix = healpy.ang2pix(self.n_side,pdfs['data_raw'][1],pdfs['data_raw'][0])
            pdf_hist = dashi.histogram.hist1d(np.linspace(-0.5,n_pix-0.5,n_pix+1))
            pdf_hist.fill(pix)
            pdfs['binned_data'] = np.array(pdf_hist.bincontent,dtype=np.uint64)
            self.binned_data = pdfs['binned_data']
            self.newdata = False
        else:
            pdfs['binned_data'] = self.binned_data
        
        #scramble data
        pix = healpy.ang2pix(self.n_side,self.data[1],np.random.uniform(0,2,len(self.data[1]))*np.pi)
        pdf_scr_hist = dashi.histogram.hist1d(np.linspace(-0.5,n_pix-0.5,n_pix+1))
        pdf_scr_hist.fill(pix)

        pdfs['binned_scr_data'] = np.array(pdf_scr_hist.bincontent,dtype=np.uint64)    
        return pdfs
        

def bg_lin_slope(x):
    ''' A simple linear background model in declination
    '''
    return (2*x+1)*np.sin(np.pi-x)

    
def bg_fisher(x):
    ''' A background model in declination based 
        on a fisher distribution with a 60 deg psf 
    '''
    return projected_fisher(60*np.pi/180,0.001,np.pi-x)*np.sin(np.pi-x)

bg_models = {'linear_slope':bg_lin_slope,'fisher60':bg_fisher}


if (__name__ == "__main__"):
    import sys
    import dashi
    import matplotlib.pyplot as plt
    dashi.visual()

    seed = int(sys.argv[1])

    N = 7.5e4 #number of events in sample
    n_side = 256 #A number on the form 2^N which sets the number of healpix bins  
    bmodel = bg_models['linear_slope'] #The background model 
    source_ext = 2#Source extension in deg
    source_dec = -50 #Source declination in deg
    nsources = 1900#Number of sources (only valid for complex source)
    bg = bg_model(N,bmodel,seed =seed)
    sig = sig_model(source_ext*np.pi/180,(source_dec*np.pi/180,266*np.pi/180),n_side = n_side,seed = seed)
    #sig = complexsource(source_ext*np.pi/180,n_side = n_side,nsources=nsources,seed = seed)
    

    selection = ToySimulation(bg,sig,n_side = n_side)
    selection.generate_data_sample(int(0.00*N))
    pdfs = selection.generate_pdfs()
    

    
    plt.figure()
    x = np.linspace(0,np.pi,100)
    plt.plot(x,bg_lin_slope(x))
    plt.figure()
    h = dashi.histogram.hist1d(np.linspace(0,np.pi,100))
    h.fill(bg.z(100000))
    h.line()
    healpy.mollview(pdfs['signal_pdf'],title='signal pdf')
    healpy.mollview(pdfs['signal_scr_pdf'],title='scrambled signal')
    healpy.mollview(pdfs['data_scr_pdf'])
    healpy.mollview(pdfs['binned_scr_data'])
    healpy.mollview(pdfs['binned_data'])
    selection.generate_data_sample(int(0.1*N))
    pdfs = selection.generate_pdfs()
    healpy.mollview(pdfs['binned_data'])
    plt.figure()
    plt.plot(pdfs['signal_pdf']/np.sum(pdfs['signal_pdf']))
    plt.plot(pdfs['data_scr_pdf']/np.sum(pdfs['data_scr_pdf']))
    plt.plot(pdfs['binned_data']/float(np.sum(pdfs['binned_data'])))
    plt.plot(pdfs['signal_scr_pdf']/np.sum(pdfs['signal_scr_pdf']))

    healpy.mollview(pdfs['data_scr_pdf']/np.sum(pdfs['data_scr_pdf']),title='data pdf')
    healpy.mollview(pdfs['data_scr_pdf']/np.sum(pdfs['data_scr_pdf'])-0.1*pdfs['signal_scr_pdf']/np.sum(pdfs['signal_scr_pdf']),title='corrected pdf')
    plt.show()


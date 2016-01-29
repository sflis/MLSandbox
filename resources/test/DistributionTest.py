#!/usr/bin/env python
from icecube.MLSandboxPythonAccess import Distribution
import unittest
#This test should check the operation of the discriminator function of the DOM.
#As of now only one test is implemented that check if the discriminator produces (doesn't produce)
#a DOMTrigger when the MCPulse has a charge bigger (smaller) than the discriminator threshold.
class SanityCheck(unittest.TestCase):


    def __init__(self):
          super(SanityCheck,self)

    def testDistributionRange(self):
        import numpy as np
        from scipy import stats
        nbins= 10
        bw2 = (10.0 / (nbins))/2.0
        x = np.linspace(0, 10-bw2*2, nbins)
        print((x[1]-x[0])/2,bw2)
        print(x[0],x[-1])
        print(bw2*2)
        #Our background is an exponential
        #bg = np.ones(bins)*stats.expon.pdf(x,scale = 5)
        #Our signal is a gaussian
        x = x+bw2
        distrpdf = stats.norm.pdf(x, loc = 5, scale=5.51)
        #x = x-bw2
        cdf_max = list()
        cdf_min = list()
        cdf_mean = list()

        pdf_max = list()
        pdf_min = list()
        pdf_mean = list()
        N = 15900
        for j in range(1):
            distr = Distribution(distrpdf,x[0],x[-1], j)
            ev = list()
            distr.SetCDFSampling(True)
            for i in range(N):
                ev.append(distr.Sample())
            print(min(ev),max(ev),np.mean(ev))
            cdf_max.append(max(ev))
            cdf_min.append(min(ev))
            cdf_mean.append(np.mean(ev))
            ev = list()
            distr.SetCDFSampling(False)
            for i in range(N):
                ev.append(distr.Sample())
            print(min(ev),max(ev),np.mean(ev))
            pdf_max.append(max(ev))
            pdf_min.append(min(ev))
            pdf_mean.append(np.mean(ev))
        print(np.mean(cdf_min),np.mean(cdf_max),np.mean(cdf_mean))
        print(np.mean(pdf_min),np.mean(pdf_max),np.mean(cdf_mean))


        N = 10590000
        import matplotlib.pyplot as plt
        distrpdf = distrpdf/sum(distrpdf)
        #x = x+bw2
        plt.plot(x,distrpdf/sum(distrpdf),'rv')
        ev = list()
        distr.SetCDFSampling(False)
        for i in range(N):
            ev.append(distr.Sample())

        hist, bins = np.histogram(np.array(ev), len(x), (0, 10))
        bincent = bins[:-1] + np.diff(bins)/2.0
        plt.plot(bincent, hist/float(N), 'go')
        print("SUM",sum(distrpdf))
        print("SUM",sum(hist/float(N)))

        print(len(bins),len(x))
        ev = list()
        distr.SetCDFSampling(True)
        for i in range(N):
            ev.append(distr.Sample())
        hist, bins = np.histogram(np.array(ev), len(x), (0, 10))
        bincent = bins[:-1] + np.diff(bins)/2.0
        plt.plot(bincent, hist/float(N), 'b+')
        plt.show()
if (__name__ == "__main__"):
    s = SanityCheck()
    s.testDistributionRange()

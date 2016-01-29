//  Copyright (C) 2012
//  Samuel Flis <samuel.flis@fysik.su.se>
//  and the IceCube Collaboration <http://www.icecube.wisc.edu>
//
//  This file is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>


#ifndef FELDMAN_COUSINS_LIKELIHOOD_H
#define FELDMAN_COUSINS_LIKELIHOOD_H

#include "Distribution.h"
#include "Histogram1D.h"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include <inttypes.h>
#include <vector>
#include <string>
#include <math.h>
#include <iostream>
uint32_t SuperFastHash (const char * data, int len);
typedef double(*likelihoodCallback)(double,void*);
/**class: Likelihood
 * \brief A base class (wrapper class) for likelihoods which defines the interface needed
 *        by FeldmanCousinsAnalysis class.
 */
class Likelihood{
    public:
        ///Evaluates the log likelihood sum
        ///\param xi the signal fraction for which the likelihood should be evaulated.
        virtual double EvaluateLLH(double xi)const = 0;

        ///Creates internally a new sample with the specified parameter likelihood parameter
        ///\param xi value of the likelihood parameter
        virtual void SampleEvents(double xi) = 0;

        ///Return the callback function of the likelihood for the minimizer
        ///\return A function pointer function of the type double(double,void*)
        virtual likelihoodCallback CallBackFcn() = 0;

        ///Should create a copy of the original Likelihood which is thread
        ///safe.
        virtual Likelihood * Clone(int seed)const = 0;

        virtual bool Changed(){
            bool ret = (stateHash_ == ChangedHash());
            stateHash_ = ChangedHash();
            return (ret || changed_);
        }

        //{return changed_; changed_=false;}

        virtual uint32_t ChangedHash() =  0;
        /// Number of events in the current sample
        uint64_t totEvents_;
        ///Expected number of events in the sample
        double N_;
    protected:
        bool changed_;
        uint32_t stateHash_;

};

class BinnedLikelihood: public Likelihood{
public:

        BinnedLikelihood(int seed, uint64_t pdfBins):
        nPDFBins_(pdfBins),
        nTotalLLHEvaluations_(0),
        eventsHistogramed_(false),
        histogramEvents_(false)
        {
            rng_ = boost::shared_ptr<RNG>(new RNG(seed));
            changed_ = true;
            totEvents_ = 0;
        }

        ///Evaluates the log likelihood
        ///\param xi the likelihood parameter for which the likelihood should be evaulated.
        virtual double EvaluateLLH(double xi)const = 0;

        ///Returns the number of likelihood evaluations.
        double GetNTotalLLHEv()const{return nTotalLLHEvaluations_;}

        ///Creates internally a new sample with the specified parameter
        virtual void SampleEvents(double xi) = 0;

        ///Return the callback function of the likelihood for the minimizer
        virtual likelihoodCallback CallBackFcn() = 0;

        ///Enables histogramming of events (must set the pdfBins_ variable)
        virtual bool EnableHistogramedEvents(){histogramEvents_ = false; return false;}

        /// This function is used to set the event sample for the likelihood to be computed.
        /// \param events a vector containing the event sample that the analysis should run on.
        void SetEvents(std::vector<double> &events){
            events_ = events;
            totEvents_ = events_.size();
            HistogramEvents();
            changed_ = true;
        }

        std::vector<double>& GetEventSample() {return events_;}
        std::vector<uint64_t>&  GetHistogramedEventSample() {return histEvents_.GetHistogramArray();}
        uint32_t ChangedHash(){return SuperFastHash((const char*) &histEvents_.GetHistogramArray(),sizeof(uint64_t)*histEvents_.GetNBins()); }
protected:
    void HistogramEvents();

    boost::shared_ptr<RNG> rng_;

    std::vector<double> events_;
    Histogram1D<uint64_t> histEvents_;

    uint64_t nPDFBins_;

    mutable uint64_t nTotalLLHEvaluations_;
    bool eventsHistogramed_;
    bool histogramEvents_;
    std::vector<uint64_t> usedBins_;
};


/**class: ShapeLikelihood
*\brief
*
*
* A class that provides functions to compute a 1 dimensional Likelihood of the form:
* \f[
    L(\xi)  = \prod^N_{k=1} (\xi f_{sig}(x_k) + (1-\xi) f_{bg}(x_k))
  \f]

* Table containing abbreviations used for variables and explenations in the comments: \n
* N: number of observed events \n
* xi = \f$ \xi = \mu / N  \f$: signal fraction \n
* x: pdf observable \n
**/
class ShapeLikelihood: public BinnedLikelihood{
    public:

        ///The SimpleShapeLikelihood constructor
        ///\param signal signal expectation (pdf)
        ///\param background background expectation (pdf)
        ///\param signalSample
        ///\param background
        ///\param N number of events in sample
        ///\param seed random number generator seed
        ShapeLikelihood(const Distribution &signal, //Signal expectation
                        const Distribution &background, //background expectation
                        const Distribution &signalSample, //Signal sample
                        const Distribution &backgroundSample, //background sample
                        double N,//
                        bool histograming = false,
                        bool poissonSampling = false,
                        int seed = 1
        ):
        BinnedLikelihood(seed, signal.GetNBins()),
        signalPdf_(signal,boost::shared_ptr<RNG>(rng_)),
        bgPdf_(background,boost::shared_ptr<RNG>(rng_)),
        signalSample_(signalSample,boost::shared_ptr<RNG>(rng_)),
        backgroundSample_(backgroundSample,boost::shared_ptr<RNG>(rng_)),
        mixed_(backgroundSample,boost::shared_ptr<RNG>(rng_)),
        poissonSampling_(poissonSampling)
        {
            histogramEvents_ = histograming;
            N_ = N;
            totEvents_ = N_;

        }




        ///The SimpleShapeLikelihood constructor
        ///\param signal signal expectation (pdf)
        ///\param background background expectation (pdf)
        ///\param signalSample
        ///\param background
        ///\param N number of events in sample
        ///\param seed random number generator seed
        ShapeLikelihood(const Distribution &signal, //Signal expectation
                        const Distribution &background, //background expectation
                        const Distribution &signalSample, //Signal sample
                        const Distribution &backgroundSample, //background sample
                        double N,//
                        int seed = 1
        ):
        BinnedLikelihood(seed, signal.GetNBins()),
        signalPdf_(signal,boost::shared_ptr<RNG>(rng_)),
        bgPdf_(background,boost::shared_ptr<RNG>(rng_)),
        signalSample_(signalSample,boost::shared_ptr<RNG>(rng_)),
        backgroundSample_(backgroundSample,boost::shared_ptr<RNG>(rng_)),
        mixed_(backgroundSample,boost::shared_ptr<RNG>(rng_)),
        poissonSampling_(false)
        {
            histogramEvents_ = false;
            N_ = N;
            totEvents_ = N_;
        }

        ///Evaluates the log likelihood sum
        ///\param xi the signal fraction for which the likelihood should be evaulated.
        double EvaluateLLH(double xi)const;

        void SampleEvents(double xi);
        likelihoodCallback CallBackFcn(){return &likelihoodEval;}

        bool EnableHistogramedEvents();
        void EnablePoissonSampling(){poissonSampling_ = true;}
        ///Signal fraction to number of events
        ///\param xi signal fraction
        ///\return number of events
        double Xi2Mu(double xi) const{
            return xi*N_;
        }

        ///Number of signal events to signal fraction
        ///\param mu number of signal events
        ///\return signal fraction
        double Mu2Xi(double mu) const{
            return mu/N_;
        }

        ShapeLikelihood * Clone(int seed)const{
            return new ShapeLikelihood(*this,
                                       seed);
        }

    private:

        ///private constructor for the clone method
        ShapeLikelihood(const ShapeLikelihood &base,int seed):
            BinnedLikelihood(seed, base.signalPdf_.GetNBins()),
            signalPdf_(base.signalPdf_,boost::shared_ptr<RNG>(base.rng_)),
            bgPdf_(base.bgPdf_,boost::shared_ptr<RNG>(base.rng_)),
            signalSample_(base.signalSample_,boost::shared_ptr<RNG>(base.rng_)),
            backgroundSample_(base.backgroundSample_,boost::shared_ptr<RNG>(base.rng_)),
            mixed_(base.mixed_,boost::shared_ptr<RNG>(base.rng_)),
            poissonSampling_(base.poissonSampling_){

                histEvents_ = base.histEvents_;
                histogramEvents_ = base.histogramEvents_;
                N_ = base.N_;
                totEvents_ = N_;
            }

        static double likelihoodEval(double xi, void *params);
        /// Distribution describing the signal pdf.
        Distribution signalPdf_;
        /// Distribution describing the background pdf.
        Distribution bgPdf_;
        Distribution signalSample_;
        Distribution backgroundSample_; //background sample
        Distribution mixed_;

        bool poissonSampling_;
};

class SignalContaminatedLH : public BinnedLikelihood{
    public:
    enum Model{None,Poisson,Binomial};

    SignalContaminatedLH(const Distribution &signal, //Signal expectation
                         const Distribution &background, //background expectation
                         const Distribution &signalScrambled, //Scrambled signal expectation
                         const Distribution &signalSample, //Signal sample
                         const Distribution &backgroundSample, //background sample
                         const Distribution &signalScrambledSample, //Scrambled signal sample
                         double N,
                         double sig_prob = 1.0,
                         double bg_prob = 1.0,
                         SignalContaminatedLH::Model model = SignalContaminatedLH::None,
                         double sig_sample_prob = 1.0,
                         double bg_sample_prob = 1.0,
                         int seed = 1
                     );


        ///Evaluates the log likelihood sum
        ///\param xi the signal fraction for which the likelihood should be evaulated.
        double EvaluateLLH(double xi)const;

        Model GetModel(){return usedModel_;}

        const Distribution& GetSignalPDF()const {return signalPdf_;}
        const Distribution& GetBgPDF()const {return bgPdf_;}

        void SampleEvents(double xi);
        bool EnableHistogramedEvents();

        uint64_t GetNEvents(){return totEvents_;}


        double Xi2W(double xi) const {
            return sig_prob_ * xi / (sig_prob_ * xi + bg_prob_*(1.0 - xi));
        }

        double W2Xi(double w) const {
            return w * bg_prob_ / (sig_prob_ + w*(bg_prob_ - sig_prob_));
        }

        double W2Mu(double w) const {
            return Xi2Mu(W2Xi(w));
        }

        double Xi2Mu(double xi) const{
            return xi*N_;
        }

        double Mu2Xi(double mu) const{
            return mu/N_;
        }

        likelihoodCallback CallBackFcn(){return &likelihoodEval;}
        SignalContaminatedLH * Clone(int seed) const {
            return new SignalContaminatedLH(signalPdf_,
                                            bgPdf_,
                                            signalPdfScrambled_,
                                            signalSample_,
                                            backgroundSample_,
                                            signalScrambledSample_,
                                            N_,
                                            sig_prob_,
                                            bg_prob_,
                                            usedModel_,
                                            sig_sample_prob_,
                                            bg_sample_prob_,
                                            seed);
        }
        uint32_t ChangedHash(){return SuperFastHash((const char*) &histEvents_.GetHistogramArray(),sizeof(uint64_t)*histEvents_.GetNBins()); }
    private:
        ///Callback function for the minimizer
        static double likelihoodEval(double xi, void *params);

        Distribution signalPdf_;
        /// Pointer to a distribution describing the signal pdf.
        Distribution signalPdfScrambled_;

        /// Pointer to a distribution describing the background pdf.
        Distribution bgPdf_;

        /// Pointer to a distribution describing the background pdf.
        Distribution bgPdfOriginal_;
        Distribution signalSample_;
        Distribution backgroundSample_; //background sample
        Distribution signalScrambledSample_; //Scrambled signal sample
        Distribution mixed_;
        Model usedModel_;
        double N_;

        double sig_prob_;
        double bg_prob_;

        double sig_sample_prob_;
        double bg_sample_prob_;

};


#include "stdint.h" /* Replace with <stdint.h> if appropriate */
#undef get16bits
#if (defined(__GNUC__) && defined(__i386__)) || defined(__WATCOMC__) \
  || defined(_MSC_VER) || defined (__BORLANDC__) || defined (__TURBOC__)
#define get16bits(d) (*((const uint16_t *) (d)))
#endif

#if !defined (get16bits)
#define get16bits(d) ((((uint32_t)(((const uint8_t *)(d))[1])) << 8)\
                       +(uint32_t)(((const uint8_t *)(d))[0]) )
#endif




#endif

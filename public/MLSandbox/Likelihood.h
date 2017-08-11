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


#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include <inttypes.h>
#include <vector>
#include <string>
#include <math.h>
#include <iostream>

#include <numeric>

class Minimizer;

uint32_t SuperFastHash (const char * data, int len);
typedef double(*likelihoodCallback)(double,void*);
/**class: Likelihood
 * \brief A base class (wrapper class) for likelihoods which defines a general interface needed
 *        analysis classes.
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

        virtual void MinimizerConditions(Minimizer &min){}
        
        virtual bool Changed(){
            bool ret = (stateHash_ == StateHash());
            stateHash_ = StateHash();
            return (ret || changed_);
        }

        virtual double MinXiBound(){return 0.0;}
        virtual double MaxXiBound(){return 1.0;}

        virtual uint32_t StateHash() =  0;
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
        nTotalLLHEvaluations_(0)
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

        virtual double MinXiBound(){return 0.0;}    
        virtual double MaxXiBound(){return 1.0;}

        /// This function is used to set the event sample for the likelihood to be computed.
        /// \param events a vector containing the binned event sample that the analysis should run on.
        void SetEvents(std::vector<uint64_t> &events){

            observation_ = events;
            if(observation_.size()!= nPDFBins_)
                throw std::invalid_argument("Event histogram has the wrong length");

    
            totEvents_ =0;
            usedBins_.clear();
            for(uint64_t i = 0; i<observation_.size();i++){
                totEvents_ += observation_[i];
                if(observation_[i]>0)
                    usedBins_.push_back(i);
            }

            //std::accumulate(observation_.begin(), observation_.end(), 0);
            
            //totEvents_ = events.size();
            changed_ = true;
        }

        std::vector<uint64_t>& GetEventSample() {return observation_;}
        uint32_t StateHash(){return SuperFastHash((const char*) &observation_[0],sizeof(uint64_t)*observation_.size()); }
protected:
    void HistogramEvents();

    boost::shared_ptr<RNG> rng_;

    std::vector<uint64_t> observation_;
    uint64_t nPDFBins_;

    mutable uint64_t nTotalLLHEvaluations_;

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
            N_ = N;
            totEvents_ = N_;
            observation_.resize(signalPdf_.GetNBins());
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
            N_ = N;
            totEvents_ = N_;
            observation_.resize(signalPdf_.GetNBins());
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

                N_ = base.N_;
                totEvents_ = N_;
                observation_ = base.observation_;
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

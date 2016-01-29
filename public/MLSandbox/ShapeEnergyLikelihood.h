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


#ifndef MLSANDBOX_SHAPE_ENERGY_LIKELIHOOD_H
#define MLSANDBOX_SHAPE_ENERGY_LIKELIHOOD_H




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
class ShapeEnergyLikelihood: public BinnedLikelihood{
    public:

        ///The SimpleShapeLikelihood constructor
        ///\param signal signal expectation (pdf)
        ///\param background background expectation (pdf)
        ///\param signalSample
        ///\param background
        ///\param N number of events in sample
        ///\param seed random number generator seed
        ShapeEnergyLikelihood(const Distribution &signal, //Signal expectation
                        const Distribution &signalEnergy, //Signal expectation
                        const Distribution &background, //background expectation
                        const Distribution &backgroundEnergy, //background expectation
                        const Distribution &signalSample, //Signal sample
                        const Distribution &signalEnergySample, //Signal sample
                        const Distribution &backgroundSample, //background sample
                        const Distribution &backgroundEnergySample, //background sample
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
        ShapeEnergyLikelihood(const Distribution &signal, //Signal expectation
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
        ShapeEnergyLikelihood(const ShapeLikelihood &base,int seed):
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
        /// Distribution describing the signal pdf.
        Distribution signalEnergyPdf_;
        /// Distribution describing the background pdf.
        Distribution bgPdf_;
        /// Distribution describing the background pdf.
        Distribution bgEnergyPdf_;

        Distribution signalSample_;
        Distribution backgroundSample_; //background sample
        Distribution signalEnergySample_;
        Distribution backgroundEnergySample_; //background sample


        Distribution mixed_;
        Distribution mixedEnergy_;
        bool poissonSampling_;
};


#endif

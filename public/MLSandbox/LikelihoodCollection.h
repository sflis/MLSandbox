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


#ifndef MLSANDBOX_LIKELIHOODCOLLECTION_H
#define MLSANDBOX_LIKELIHOODCOLLECTION_H
#include "MLSandbox/Likelihood.h"
#include <iostream>
#include <string>
#include <map>

class Minimizer;

class LikelihoodCollection : public BinnedLikelihood{
    public:
    enum Model{None,Poisson,Binomial};

    LikelihoodCollection(const Distribution &signal, //Signal expectation
                         const Distribution &background, //background expectation
                         const Distribution &signalScrambled, //Scrambled signal expectation
                         const Distribution &signalSample, //Signal sample
                         const Distribution &backgroundSample, //background sample
                         const Distribution &signalScrambledSample, //Scrambled signal sample
                         double N,
                         double sig_prob = 1.0,
                         double bg_prob = 1.0,
                         LikelihoodCollection::Model model = LikelihoodCollection::None,
                         double sig_sample_prob = 1.0,
                         double bg_sample_prob = 1.0,
                         int seed = 1
                     );


        ///Evaluates the log likelihood sum
        ///\param xi the signal fraction for which the likelihood should be evaulated.
        double EvaluateLLH(double xi)const;


        double NoSigSubCorr(double xi)const;

        double StandardSigSub(double xi)const;

        // double NonTerminatedSigSub(double xi)const;

        // double HybridSigSub(double xi)const;


        Model GetModel(){return usedModel_;}

        const Distribution& GetSignalPDF()const {return signalPdf_;}
        const Distribution& GetBgPDF()const {return bgPdf_;}

        void SampleEvents(double xi);

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
        
        void MinimizerConditions(Minimizer &min);

        likelihoodCallback CallBackFcn(){return &likelihoodEval;}
        LikelihoodCollection * Clone(int seed) const {
            return new LikelihoodCollection(signalPdf_,
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

        double MaxXiBound(){return  maxSFractionFit_;}

        void SetLLHFunction2(double (*fcpt)(const LikelihoodCollection &, double) ){current_llh_ = fcpt; changed_ = true; }
        void SetLLHFunction(std::string fc_name);

        static double noSigSubCorr(const LikelihoodCollection &likelihood, double xi);

        static double standardSigSub(const LikelihoodCollection & likelihood, double xi);

        // static double nonTerminatedSigSub(LikelihoodCollection & likelihood, double xi);

        // static double hybridSigSub(LikelihoodCollection & likelihood, double xi);

        double (*current_llh_)(const LikelihoodCollection & , double);

    private:
        ///Callback function for the minimizer
        static double likelihoodEval(double xi, void *params);
        

        void ComputeMaxSFrac();
        std::map<std::string, double (*)(const LikelihoodCollection & , double)> callbackMap_;
        
        
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

        double sig_prob_;
        double bg_prob_;

        double sig_sample_prob_;
        double bg_sample_prob_;

        double maxSFractionFit_;

        double lastInjXi_;
};

#endif

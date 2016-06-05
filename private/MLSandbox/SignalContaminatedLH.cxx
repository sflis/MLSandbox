#include "MLSandbox/Likelihood.h"
#include "MLSandbox/SignalContaminatedLH.h"
#include "MLSandbox/Minimizer.h"
#include <math.h>
#include <iostream>
#include <cfloat>
#include <numeric>
#include <stdexcept>
#include <limits>
using namespace std;


//_____________________________________________________________________________
SignalContaminatedLH::SignalContaminatedLH(const Distribution &signal, //Signal expectation
                     const Distribution &background, //background expectation
                     const Distribution &signalScrambled, //Scrambled signal expectation
                     const Distribution &signalSample, //Signal sample
                     const Distribution &backgroundSample, //background sample
                     const Distribution &signalScrambledSample, //Scrambled signal sample
                     double N,
                     double sig_prob ,
                     double bg_prob,
                     SignalContaminatedLH::Model model,
                     double sig_sample_prob,
                     double bg_sample_prob,
                     int seed
                     ):
            BinnedLikelihood(seed,signal.GetNBins()),
            signalPdf_(signal,boost::shared_ptr<RNG>(rng_)),
            signalPdfScrambled_(signalScrambled,boost::shared_ptr<RNG>(rng_)),
            bgPdf_(background,boost::shared_ptr<RNG>(rng_)),
            bgPdfOriginal_(background,boost::shared_ptr<RNG>(rng_)),
            signalSample_(signalSample,boost::shared_ptr<RNG>(rng_)),
            backgroundSample_(backgroundSample,boost::shared_ptr<RNG>(rng_)),
            signalScrambledSample_(signalScrambledSample,boost::shared_ptr<RNG>(rng_)),
            mixed_(background,boost::shared_ptr<RNG>(rng_)),
            usedModel_(model),
            sig_prob_(sig_prob),
            bg_prob_(bg_prob),
            sig_sample_prob_(sig_sample_prob),
            bg_sample_prob_(bg_sample_prob){
                N_ = N;
                if(usedModel_ == Binomial){
                    double p = sig_prob_*0.5 + bg_prob_*(1 - 0.5);
                    double epsilon = 1e-21;
                    if(p<epsilon or (1-p)< epsilon){
                        throw std::invalid_argument("Binomial model invalid with given signal and background probabilities. Try modle `None'.");
                    }
                }
                observation_.resize(signalPdf_.GetNBins());
                ComputeMaxSFrac();
            }

//_____________________________________________________________________________
double SignalContaminatedLH::EvaluateLLH(double xi) const{
    double llhSum = 0;
    double illhSum = 0;
    const double w = Xi2W(xi);

    // Loop over the binned events to evaluate the likelihood.
    for (std::vector<uint64_t>::const_iterator it=usedBins_.begin(); it!=usedBins_.end(); ++it){
        uint64_t index = *it;
        double bg_prob = bgPdf_[index] - xi*signalPdfScrambled_[index];
        //if(bg_prob<0)
        //    bg_prob = 0;
        double t_prob = w * signalPdf_[index] + (1-w)/(1-xi)*( bg_prob);
        //t_prob = (t_prob+1)/(-(1e6-2)*t_prob+1e6)*fabs(t_prob); 
        //cout<<t_prob<<endl;
        if(t_prob<=0){
            //cout<<w<<" "<<xi<<" "<<bg_prob<<" "<<t_prob<<endl;
            t_prob = std::numeric_limits<double>::min();
        }
        /*
        if(t_prob<0){
            llhSum += observation_[index]
            * log(-t_prob);
            illhSum += M_PI/2 +log(1-t_prob);// 1-w;
        }
        else{//*/    
            llhSum += observation_[index]
            * log(t_prob);
        //}
    }

    // Adding poisson or binomial factor to the likelihood if enabled.
    switch(usedModel_){
        case Poisson:
            llhSum += -(N_*(xi*sig_prob_ + (1 - xi)*bg_prob_)) +
            totEvents_*log(N_*(xi*sig_prob_ + (1 - xi)*bg_prob_));
            break;
        case Binomial:
        {
            double p = sig_prob_*xi + bg_prob_*(1 - xi);
            llhSum += log(gsl_ran_binomial_pdf(totEvents_, p, N_));
        }
            break;
        case None:
        default:
            break;
    }
    // Counting the number of llh evaluations.
    nTotalLLHEvaluations_++;

    return llhSum;//-sqrt(llhSum*llhSum+illhSum*illhSum);
}
//_____________________________________________________________________________
void SignalContaminatedLH::SampleEvents(double xi){
    double injectedSignal = Xi2Mu(xi);
    double w = Xi2W(xi);
    if(xi!= lastInjXi_){
        //FIXME: probably wrong to use backgroundSample_ to create new bgPdf_
        addDistributions(xi, signalScrambledSample_, 1-xi, backgroundSample_, bgPdf_);
        addDistributions(w, signalSample_, - xi*(1-w)/(1-xi), signalScrambledSample_, mixed_);
        addDistributions(1.0, mixed_, (1-w)/(1-xi), backgroundSample_, mixed_);
        lastInjXi_ = xi;
    }
    std::vector<double> &s  = mixed_.GetPDFVector();
    if(xi<0 or xi>1.0){
        throw std::invalid_argument("Signal fraction xi out of bounds [0,1]");
    }
    usedBins_.clear();
    switch(usedModel_){
        case Poisson:
        {
           uint64_t current_n = rng_->Poisson(N_ * ((1 - xi) * bg_sample_prob_ + xi * sig_sample_prob_));
           std::fill(observation_.begin(), observation_.end(), 0);
           for(uint64_t j = 0; j < current_n; j++){
               uint64_t i = mixed_.SampleFromDistrI();
               observation_[i] +=1;
               std::vector<uint64_t>::iterator  it = std::lower_bound(usedBins_.begin(), usedBins_.end(), i);
               if(it == usedBins_.end() || i < *it){
                   usedBins_.insert(it,i);
               }
           }

        }
            break;
        case Binomial:
        {
           double p = sig_prob_*xi + bg_prob_*(1 - xi);
           uint64_t current_bg = rng_->Binomial(bg_sample_prob_*(1 - xi), N_);
           uint64_t current_mu = rng_->Binomial(sig_sample_prob_*xi, N_);
           std::fill(observation_.begin(), observation_.end(), 0);
           for(uint64_t j = 0; j < current_bg + current_mu; j++){
               uint64_t i = mixed_.SampleFromDistrI();
               observation_[i] +=1;
               std::vector<uint64_t>::iterator  it = std::lower_bound(usedBins_.begin(), usedBins_.end(), i);
               if(it == usedBins_.end() || i < *it){
                   usedBins_.insert(it,i);
               }
           }
        }
            break;
        case None:
        default:
        {
            totEvents_ = rng_->Poisson(N_);
            //*
            std::vector<double> &pdf =  mixed_.GetPDFVector();
            for(uint64_t i = 0, n = pdf.size(); i<n; i++){
                uint64_t events = rng_->Poisson(pdf[i]*N_);
                observation_[i] = events;
                if(events!=0){
                    usedBins_.push_back(i);
                }
            }

            /*/
            std::fill(observation_.begin(), observation_.end(), 0);
            for(uint64_t j = 0; j < totEvents_; j++){
                uint64_t i = mixed_.SampleFromDistrI();
                observation_[i] +=1;
                std::vector<uint64_t>::iterator  it = std::lower_bound(usedBins_.begin(), usedBins_.end(), i);
                if(it == usedBins_.end() || i < *it){
                    usedBins_.insert(it,i);
                }
            }
            //*/
        }
            break;
    }
    //should be more efficient to sum this up while filling the bins
    totEvents_ = std::accumulate(observation_.begin(), observation_.end(), 0);
    ComputeMaxSFrac();
    changed_ = true;
}
//_____________________________________________________________________________
void SignalContaminatedLH::MinimizerConditions(Minimizer &min){
     min.SetBoundaries(0.0,maxSFractionFit_);
}
//_____________________________________________________________________________
void SignalContaminatedLH::ComputeMaxSFrac(){
    maxSFractionFit_ = N_;
    for(uint64_t i = 0; i<signalPdf_.GetNBins(); i++){
        if(maxSFractionFit_< bgPdf_[i]/(signalPdfScrambled_[i]-signalPdf_[i]))
            maxSFractionFit_ = bgPdf_[i]/(signalPdfScrambled_[i]-signalPdf_[i]);
    }
}
//_____________________________________________________________________________
double SignalContaminatedLH::likelihoodEval(double xi, void *params){
    return -((SignalContaminatedLH*) params)->EvaluateLLH(xi);
}

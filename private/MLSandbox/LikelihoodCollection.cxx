
#include "MLSandbox/LikelihoodCollection.h"
#include "MLSandbox/Minimizer.h"
#include <math.h>
#include <iostream>
#include <cfloat>
#include <numeric>
#include <stdexcept>
#include <limits>
using namespace std;


//_____________________________________________________________________________
LikelihoodCollection::LikelihoodCollection(const Distribution &signal, //Signal expectation
                     const Distribution &background, //background expectation
                     const Distribution &signalScrambled, //Scrambled signal expectation
                     const Distribution &signalSample, //Signal sample
                     const Distribution &backgroundSample, //background sample
                     const Distribution &signalScrambledSample, //Scrambled signal sample
                     double N,
                     double sig_prob ,
                     double bg_prob,
                     LikelihoodCollection::Model model,
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
            bg_sample_prob_(bg_sample_prob),
            lastInjXi_(std::numeric_limits<double>::quiet_NaN()){
                N_ = N;
                if(usedModel_ == Binomial){
                    double p = sig_prob_*0.5 + bg_prob_*(1 - 0.5);
                    double epsilon = 1e-21;
                    if(p<epsilon or (1-p)< epsilon){
                        throw std::invalid_argument("Binomial model invalid with given signal and background probabilities. Try model `None'.");
                    }
                }
                observation_.resize(signalPdf_.GetNBins());
                ComputeMaxSFrac();
                current_llh_ = standardSigSub;
                callbackMap_["standardSigSub"] = standardSigSub;
                callbackMap_["noSigSubCorr"] = noSigSubCorr;
            }

//_____________________________________________________________________________
double LikelihoodCollection::EvaluateLLH(double xi) const{
    return current_llh_(*this,xi);
}
//_____________________________________________________________________________
double LikelihoodCollection::StandardSigSub(double xi)const{
    double llhSum = 0;
     //Just in case to avert
    //NaNs in the likelihood
    if(xi == 1)
        xi = 1.0 - std::numeric_limits<double>::epsilon();
    
    const double w = Xi2W(xi);
    const uint64_t nbins = mixed_.GetNBins();
    // Loop over the binned events to evaluate the likelihood.
    for (std::vector<uint64_t>::const_iterator it=usedBins_.begin(); it!=usedBins_.end(); ++it){
        uint64_t index = *it;
        double bg_prob = bgPdf_[index] - xi*signalPdfScrambled_[index];
        double t_prob = w * signalPdf_[index] + (1-w)/(1-xi)*( bg_prob);

        if(t_prob<=0){
            t_prob = std::numeric_limits<double>::min();
        }

        if(bg_prob<0){
            llhSum -= std::numeric_limits<double>::max()/mixed_.GetNBins();
        }
        else{    
            llhSum += observation_[index]
            * log(t_prob);
        }
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

    return llhSum;

}
//_____________________________________________________________________________
double LikelihoodCollection::NoSigSubCorr(double xi)const{
    double llhSum = 0;
    const double bgFraction = (1-xi);

    // Only loop over the bins which contain events to evaluate the likelihood.
    for (std::vector<uint64_t>::const_iterator it=usedBins_.begin(); it!=usedBins_.end(); ++it){
        uint64_t index = *it;
        llhSum += observation_[index] *
        log( xi * signalPdf_[index] + bgFraction * bgPdf_[index]);

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

    return llhSum;

}

double LikelihoodCollection::noSigSubCorr(const LikelihoodCollection &likelihood, double xi){
  return likelihood.NoSigSubCorr(xi);
}

double LikelihoodCollection::standardSigSub(const LikelihoodCollection & likelihood, double xi){
  return likelihood.StandardSigSub(xi);
}
//_____________________________________________________________________________
void LikelihoodCollection::SampleEvents(double xi){
    double injectedSignal = Xi2Mu(xi);
    double w = Xi2W(xi);


    std::vector<double> &s  = mixed_.GetPDFVector();
    if(xi<0 or xi>1.0){
        throw std::invalid_argument("Signal fraction xi out of bounds [0,1]");
    }
    
    usedBins_.clear();
    switch(usedModel_){
        case Poisson:
        {
           /*uint64_t current_n = rng_->Poisson(N_ * ((1 - xi) * bg_sample_prob_ + xi * sig_sample_prob_));
           std::fill(observation_.begin(), observation_.end(), 0);
           for(uint64_t j = 0; j < current_n; j++){
               uint64_t i = mixed_.SampleFromDistrI();
               observation_[i] +=1;
               std::vector<uint64_t>::iterator  it = std::lower_bound(usedBins_.begin(), usedBins_.end(), i);
               if(it == usedBins_.end() || i < *it){
                   usedBins_.insert(it,i);
               }
           }
          */
            if(xi != lastInjXi_){
                addDistributions(w, signalScrambledSample_, 1-w, bgPdfOriginal_, bgPdf_);
                addDistributions(xi, signalSample_,  (1-xi), backgroundSample_, mixed_);
                lastInjXi_ = xi;
           }
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
        case Binomial:
        {
           if(xi != lastInjXi_){
              
                addDistributions(xi, signalScrambledSample_, 1-xi, bgPdfOriginal_, bgPdf_);
                addDistributions(w, signalSample_, - xi*(1-w)/(1-xi), signalScrambledSample_, mixed_);
                addDistributions(1.0, mixed_, (1-w)/(1-xi), backgroundSample_, mixed_);
                lastInjXi_ = xi;
           }
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
            
            if(xi != lastInjXi_){
              //FIXME: probably wrong to use backgroundSample_ to create new bgPdf_
                addDistributions(xi, signalScrambledSample_, 1-xi, backgroundSample_, bgPdf_);
                addDistributions(w, signalSample_, - xi*(1-w)/(1-xi), signalScrambledSample_, mixed_);
                addDistributions(1.0, mixed_, (1-w)/(1-xi), backgroundSample_, mixed_);
                lastInjXi_ = xi;
           }
            totEvents_ = rng_->Poisson(N_);
            /*
            std::vector<double> &pdf =  mixed_.GetPDFVector();
            for(uint64_t i = 0, n = pdf.size(); i<n; i++){
                uint64_t events = rng_->Poisson(pdf[i]*N_);
                observation_[i] = events;
                if(events != 0){
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
void LikelihoodCollection::SetLLHFunction(std::string fc_name){ 
    auto search = callbackMap_.find(fc_name);  
    if(search != callbackMap_.end()){
        current_llh_ = search->second;
        changed_ = true;
    }
    else
        std::cout<<"Likelihood was not found"<<std::endl;
}
//_____________________________________________________________________________
void LikelihoodCollection::MinimizerConditions(Minimizer &min){
     //min.SetBoundaries(0.0,maxSFractionFit_);
}
//_____________________________________________________________________________
void LikelihoodCollection::ComputeMaxSFrac(){
    maxSFractionFit_ = 1.0;
    for(uint64_t i = 0; i<signalPdf_.GetNBins(); i++){
        if(observation_[i]>0 && maxSFractionFit_> bgPdf_[i]/(signalPdfScrambled_[i])){
            maxSFractionFit_ = bgPdf_[i]/(signalPdfScrambled_[i]);
        }
    }

}
//_____________________________________________________________________________
double LikelihoodCollection::likelihoodEval(double xi, void *params){
    return -((LikelihoodCollection*) params)->EvaluateLLH(xi);
}

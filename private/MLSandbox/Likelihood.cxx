#include "MLSandbox/Likelihood.h"
#include <math.h>
#include <iostream>
#include <cfloat>
#include <numeric>
#include <stdexcept>
using namespace std;

//_____________________________________________________________________________
double ShapeLikelihood::EvaluateLLH(double xi) const{
    double llhSum = 0;
    const double bgFraction = (1-xi);

    uint64_t n = usedBins_.size();
    // Loop over the bins of the event histogram to evaluate the likelihood.
    for (std::vector<uint64_t>::const_iterator it=usedBins_.begin(); it!=usedBins_.end(); ++it){
        uint64_t index = *it;
        llhSum += observation_[index] *
        log( xi * signalPdf_[index] +
            bgFraction * bgPdf_[index]);
    }

    // Counting the number of llh evaluations.
    nTotalLLHEvaluations_++;

    return llhSum;
}

//_____________________________________________________________________________
double ShapeLikelihood::likelihoodEval(double xi, void *params){
    return -((ShapeLikelihood*) params)->EvaluateLLH(xi);
}
//_____________________________________________________________________________
 void ShapeLikelihood::SampleEvents(double xi){
    if(xi>1){
        cerr<<"Invalid signal fraction: xi ("<<xi<<") must lie between 0<=xi<=1"<<endl;
        exit(1);
    }
    changed_ = true;
    usedBins_.clear();
    addDistributions(xi, signalSample_, 1.0 - xi, backgroundSample_, mixed_);
    totEvents_ = rng_->Poisson(N_);

    //If the number of events is larger than the number of pdf bins
    //poisson sampling is probably faster
    if(poissonSampling_){

        std::vector<double> &pdf =  mixed_.GetPDFVector();
        for(uint64_t i = 0, n = pdf.size(); i<n; i++){
            uint64_t events = rng_->Poisson(pdf[i]*N_);
            observation_[i] = events;
            if(events!=0){
                usedBins_.push_back(i);
            }
        }
    }
    else{
        std::fill(observation_.begin(), observation_.end(), 0);
        for(uint64_t j = 0; j < totEvents_; j++){
            uint64_t i = mixed_.SampleFromDistrI();
            observation_[i] +=1;
            std::vector<uint64_t>::iterator  it = std::lower_bound(usedBins_.begin(), usedBins_.end(), i);
            if(it == usedBins_.end() || i < *it){
                usedBins_.insert(it,i);
            }
        }
    }
}
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

            }

//_____________________________________________________________________________
double SignalContaminatedLH::EvaluateLLH(double xi) const{
    double llhSum = 0;
    const double bgFraction = (1-xi);
    const double w = Xi2W(xi);
    const double w_bg = 1-w;
    uint64_t n = usedBins_.size();

    // Loop over the bins of the event histogram to evaluate the likelihood.
    for (std::vector<uint64_t>::const_iterator it=usedBins_.begin(); it!=usedBins_.end(); ++it){
        uint64_t index = *it;
        llhSum += observation_[index]
        * log( w * signalPdf_[index] + bgPdf_[index] - w*signalPdfScrambled_[index] );

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
void SignalContaminatedLH::SampleEvents(double xi){
    double injectedSignal = Xi2Mu(xi);
    double w = Xi2W(xi);
    //FIXME: probably wrong to use backgroundSample_ to create new bgPdf_
    addDistributions(w, signalPdfScrambled_, 1-w, backgroundSample_, bgPdf_);
    addDistributions(w, signalSample_, - w, signalScrambledSample_, mixed_);
    addDistributions(1.0, mixed_, 1.0, backgroundSample_, mixed_);
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
           for(uint64_t j = 0; j < current_bg+current_mu; j++){
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
            //usedBins_.clear();
            /*
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

    changed_ = true;
}
//_____________________________________________________________________________
double SignalContaminatedLH::likelihoodEval(double xi, void *params){
    return -((SignalContaminatedLH*) params)->EvaluateLLH(xi);
}



uint32_t SuperFastHash (const char * data, int len) {
    uint32_t hash = len, tmp;
    int rem;

    if (len <= 0 || data == NULL) return 0;

    rem = len & 3;
    len >>= 2;

    /* Main loop */
    for (;len > 0; len--) {
        hash  += get16bits (data);
        tmp    = (get16bits (data+2) << 11) ^ hash;
        hash   = (hash << 16) ^ tmp;
        data  += 2*sizeof (uint16_t);
        hash  += hash >> 11;
    }

    /* Handle end cases */
    switch (rem) {
        case 3: hash += get16bits (data);
                hash ^= hash << 16;
                hash ^= ((signed char)data[sizeof (uint16_t)]) << 18;
                hash += hash >> 11;
                break;
        case 2: hash += get16bits (data);
                hash ^= hash << 11;
                hash += hash >> 17;
                break;
        case 1: hash += (signed char)*data;
                hash ^= hash << 10;
                hash += hash >> 1;
    }

    /* Force "avalanching" of final 127 bits */
    hash ^= hash << 3;
    hash += hash >> 5;
    hash ^= hash << 4;
    hash += hash >> 17;
    hash ^= hash << 25;
    hash += hash >> 6;

    return hash;
}

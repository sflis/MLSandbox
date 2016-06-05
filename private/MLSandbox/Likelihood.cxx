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

    // Only loop over the bins which contain events to evaluate the likelihood.
    for (std::vector<uint64_t>::const_iterator it=usedBins_.begin(); it!=usedBins_.end(); ++it){
        uint64_t index = *it;
        llhSum += observation_[index] *
        log( xi * signalPdf_[index] + bgFraction * bgPdf_[index]);

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

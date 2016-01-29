#include "MLSandbox/CombinedLikelihood.h"
#include <math.h>
#include <iostream>
#include <cfloat>
#include <numeric>
#include <stdexcept>
using namespace std;

//_____________________________________________________________________________
double CombinedLikelihood::EvaluateLLH(double xi) const{
    double llhValue=0;
    for(uint64_t i = 0; i < likelihoods_.size(); i++){
        double xi_i = xi / likelihoods_[i]->totEvents_ * weights_[i] * totEvents_;
        llhValue += likelihoods_[i]->EvaluateLLH(xi_i);
    }
    return llhValue;
}
//_____________________________________________________________________________
double CombinedLikelihood::likelihoodEval(double xi, void *params){
    return -((CombinedLikelihood*) params)->EvaluateLLH(xi);
}
//_____________________________________________________________________________
void CombinedLikelihood::SampleEvents(double xi){
    totEvents_ = 0;
    for(uint64_t i = 0; i<likelihoods_.size(); i++){
        double xi_i = xi / likelihoods_[i]->N_ * weights_[i] * N_;
        likelihoods_[i]->SampleEvents(xi_i);
        totEvents_ += likelihoods_[i]->totEvents_;
    }
    changed_ = true;
}

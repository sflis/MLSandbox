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
        //double xi_i = xi  * weights_[i];
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
        //double xi_i = xi  * weights_[i];
        likelihoods_[i]->SampleEvents(xi_i);
        totEvents_ += likelihoods_[i]->totEvents_;
    }
    changed_ = true;
}
//_____________________________________________________________________________
double CombinedLikelihood::MaxXiBound(){
    double xi =1.0;
    for(uint64_t i = 0; i < likelihoods_.size(); i++){
        double xi_ = likelihoods_[i]->MaxXiBound()*likelihoods_[i]->totEvents_ /(weights_[i] * totEvents_);
        if(xi>xi_)
            xi = xi_;
    }
    return xi;

}
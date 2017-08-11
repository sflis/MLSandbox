#include "MLSandbox/Distribution.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

#include <math.h>
#include <iostream>
#include <vector>
#include <cfloat>
#include <algorithm>
#include <numeric>
#include <limits>
using namespace std;

Distribution::Distribution(Distribution const &base, boost::shared_ptr<RNG> rng):
    rangeMax_(base.rangeMax_),
    rangeMin_(base.rangeMin_),
    useCDF_(base.useCDF_),
    pdfMax_(base.pdfMax_),
    seed_(rng->seed_),
    pdf_(base.pdf_),
    cdf_(base.cdf_),
    binWidth_(base.binWidth_),
    invBinWidth_(base.invBinWidth_),
    range_(base.range_),
    nBins_(base.nBins_),
    rng_(rng)
    {}

//_____________________________________________________________________________
Distribution::Distribution(const std::vector<double> &distribution, double rMin, double rMax, uint rSeed):
rangeMax_(rMax),
rangeMin_(rMin),
useCDF_(false),
seed_(rSeed){
    rng_ = boost::shared_ptr<RNG> (new RNG(rSeed));
    range_ = rangeMax_ - rangeMin_;
    nBins_ = distribution.size();
    if(range_ < 0){
        cerr<<"Error: interval is negative..."<<endl;
        return;
    }



    binWidth_ = range_/(nBins_);
    invBinWidth_ = 1/binWidth_;
    pdf_ = distribution;
    cdf_.resize(nBins_,0);

    range_ = rangeMax_ -binWidth_ - rangeMin_;

    double sum = std::accumulate(pdf_.begin(), pdf_.end(),0.0);
    pdf_[0] /= sum;
    cdf_[0] += pdf_[0];
    double max = -DBL_MAX;
    //Filling the cdf and normalizing the pdf.
    for(uint64_t i = 1; i < nBins_; i++){
        pdf_[i] /= sum;
        if(max < pdf_[i])
            max = pdf_[i];
        cdf_[i] = cdf_[i-1] + pdf_[i];
    }


    pdfMax_ = max;
    // double area = 0;
    // for(uint64_t i = 1; i < 10000; i++){
    //     double x = rangeMin_ + rng_->Uniform() * range_;
    //     double y = rng_->Uniform() * pdfMax;
    //     if(PDF(x) < y)
    //         area += 1;
    // }

    useCDF_ = true;

    //Special case to handle '0-probability distributions'
    if(sum == 0){

        for(uint64_t i = 1; i < nBins_; i++){
            pdf_[i] = 0;
            cdf_[i] = 0;
        }
        invBinWidth_=0;
        rangeMin_ = 1;
        rangeMax_ = 0;
        pdfMax_=-1;
        range_ = std::numeric_limits<double>::quiet_NaN();
    }
}
//_____________________________________________________________________________
double Distribution::PDF(double x)const{
   if(x >= rangeMin_ && x < rangeMax_)
      return pdf_[(x - rangeMin_) * invBinWidth_];
   else
      return 0;
}
//_____________________________________________________________________________
double Distribution::CDF(double x)const{
   if(x >= rangeMin_ && x < rangeMax_)
      return cdf_[(x - rangeMin_) * invBinWidth_];
   else if(x < rangeMin_)
      return 0;
   else // For x > rangeMax.
      return 1;
}
//_____________________________________________________________________________
double Distribution::SampleFromDistr()const{


   if(useCDF_){
        double cdf_value = rng_->Uniform();
        std::vector<double>::const_iterator up;
        up = std::upper_bound(cdf_.begin(), cdf_.end(), cdf_value);
        return  rangeMin_ + ((up - cdf_.begin()) + rng_->Uniform()*0.5 ) * binWidth_;//rng->Uniform()
   }
   double x;
   double y = 0;

    //If the inverse CDF isn't defined we use the 'accept and reject'
    //method to throw an event from the distribution PDF.
    do{
        x = rangeMin_ + rng_->Uniform() * range_;
        y = rng_->Uniform() * pdfMax_;

    }while(PDF(x) <= y);

   return x;
}


//_____________________________________________________________________________

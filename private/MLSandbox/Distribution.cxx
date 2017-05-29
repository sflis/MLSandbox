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
    rangeMax(base.rangeMax),
    rangeMin(base.rangeMin),
    cdf(base.cdf),
    pdf(base.pdf),
    useCDF_(base.useCDF_),
    pdfMax(base.pdfMax),
    seed(rng->seed_),
    pdf_(base.pdf_),
    cdf_(base.cdf_),
    binWidth_(base.binWidth_),
    invBinWidth_(base.invBinWidth_),
    range_(base.range_),
    nBins_(base.nBins_),
    rng(rng)
    {}
//_____________________________________________________________________________
Distribution::Distribution(ptDFctD pdf, double rMin, double rMax, uint64_t nBins, uint rSeed):
rangeMax(rMax),
rangeMin(rMin),
useCDF_(false),
seed(rSeed){

    rng = boost::shared_ptr<RNG> (new RNG(rSeed));

    range_ = rangeMax - rangeMin;
    nBins_ = nBins;
    if(range_ < 0){
        cerr<<"Error: interval is negative..."<<endl;
        return;
    }

    binWidth_ = range_ / nBins_;
    invBinWidth_ = 1 / binWidth_;
    range_ = rangeMax+binWidth_ - rangeMin;
    rangeMax += binWidth_;
    pdf_.resize(nBins_, 0);
    cdf_.resize(nBins_, 0);

    double sum = 0;
    //Filling the pdf
    pdf_[0] = pdf(rangeMin + (0.5)*binWidth_);
    cdf_[0] = pdf_[0];
    for(uint64_t i = 1; i < nBins_; i++){
        pdf_[i] = pdf(rangeMin + (0.5 + i) * binWidth_);
        sum += pdf_[i];
        cdf_[i] = pdf_[i] + cdf_[i-1];
    }
    double max = 0;
    for(uint64_t i = 0; i < nBins_; i++){

        pdf_[i] /= sum;
        if(max < pdf_[i])
            max = pdf_[i];
        cdf_[i] /= sum;
    }
    pdfMax = max;

    double area = 0;
    for(uint64_t i = 1; i < 10000; i++){
        double x = rangeMin + rng->Uniform() * range_;
        double y = rng->Uniform() * pdfMax;
        if(PDF(x) < y)
            area += 1;
    }
//     area /= 10000;
// //     if(nBins_*area < log(nBins_))
        useCDF_ = true;

}
//_____________________________________________________________________________
Distribution::Distribution(const std::vector<double> &distribution, double rMin, double rMax, uint rSeed):
rangeMax(rMax),
rangeMin(rMin),
useCDF_(false),
seed(rSeed){
    rng = boost::shared_ptr<RNG> (new RNG(rSeed));
    range_ = rangeMax - rangeMin;
    nBins_ = distribution.size();
    if(range_ < 0){
        cerr<<"Error: interval is negative..."<<endl;
        return;
    }



    binWidth_ = range_/(nBins_-1);
    invBinWidth_ = 1/binWidth_;
    pdf_ = distribution;
    cdf_.resize(nBins_,0);

    range_ = rangeMax+binWidth_ - rangeMin;
    rangeMax += binWidth_*.5;
    rangeMin -= binWidth_*.5;
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


    pdfMax = max;
    double area = 0;
    for(uint64_t i = 1; i < 10000; i++){
        double x = rangeMin + rng->Uniform() * range_;
        double y = rng->Uniform() * pdfMax;
        if(PDF(x) < y)
            area += 1;
    }

        useCDF_ = true;

    //Special case to handle '0-probability distributions'
    if(sum == 0){
        //cout<<"Special case"<<endl;
        for(uint64_t i = 1; i < nBins_; i++){
            pdf_[i] = 0;
            cdf_[i] = 0;
        }
        invBinWidth_=0;
        rangeMin = 1;
        rangeMax = 0;
        pdfMax=-1;
        range_ = std::numeric_limits<double>::quiet_NaN();
    }
}
//_____________________________________________________________________________
double Distribution::PDF(double x)const{
   if(x >= rangeMin && x < rangeMax)
      return pdf_[(x - rangeMin) * invBinWidth_];
   else
      return 0;
}
//_____________________________________________________________________________
double Distribution::CDF(double x)const{
   if(x >= rangeMin && x < rangeMax)
      return cdf_[(x - rangeMin) * invBinWidth_];
   else if(x < rangeMin)
      return 0;
   else // For x > rangeMax.
      return 1;
}
//_____________________________________________________________________________
double Distribution::SampleFromDistr()const{


   if(useCDF_){
        double cdf_value = rng->Uniform();
        std::vector<double>::const_iterator up;
        up = std::upper_bound(cdf_.begin(), cdf_.end(), cdf_value);
        return  rangeMin + ((up - cdf_.begin()) + rng->Uniform() ) * binWidth_;//rng->Uniform()
   }
   double x;
   double y = 0;

    //If the inverse CDF isn't defined we use the 'accept and reject'
    //method to throw an event from the distribution PDF.
    do{
        x = rangeMin + rng->Uniform() * range_;
        y = rng->Uniform() * pdfMax;

    }while(PDF(x) <= y);

   return x;
}


//_____________________________________________________________________________

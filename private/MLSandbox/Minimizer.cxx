#include <gsl/gsl_roots.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_errno.h>
#include <exception>
#include <stdexcept>
#include <stdio.h>
#include <cmath>
#include "MLSandbox/Minimizer.h"


#include <iostream>
#include <string>
using namespace std;
//_____________________________________________________________________________
double Minimizer::ComputeBestFit(Likelihood &lh){
    /// Determining the best fit (maximizing the likelihood) with the gsl library minimizer 
    /// using brent's method.
    /// domain mu=[0,nObs].
    lh.MinimizerConditions(*this);
    double searchInterval = 5e-3;
    double lPoint = 0; //Left starting point of the search interval method
    double rPoint = searchInterval;//Right starting point of the search interval
    double interval = rPoint - lPoint;
    double mPoint = interval/2 + lPoint; //Middle slope
    double f_1 = lh.EvaluateLLH(lPoint);
    double f_2 = lh.EvaluateLLH((1e-5));
    char error_str[300]; 

    // For the boundary case best fit mu = 0 the slope is always negative and we only need to compute
    // one slope.
    if( (f_2 - f_1) < 0){
        bestFit_ = 0;
        bestFitLLH_ = f_1;
        return bestFitLLH_;
    }
    //Searching for the right interval
    rPoint = searchInterval/2;
    while(rPoint < 1.0 && f_2 > f_1 && rPoint<maxXi_){
        rPoint += searchInterval;
        f_1 = f_2;
        f_2 = lh.EvaluateLLH(rPoint);
    }
    if(rPoint>1.0)
        rPoint=1.0;
    if(rPoint>maxXi_)
        rPoint=maxXi_;
    lPoint = rPoint - searchInterval*2;
    mPoint = rPoint - searchInterval;

    // Variables needed for the gsl minimizer
    int status;
    int  max_iter = 30;
    nIterations_ = 0;
    gsl_function llh;

    llh.function = lh.CallBackFcn();
    llh.params = &lh;

    
    // Since we cannot garantee f(a)>f(x)<f(b) for all x in [a,b] we check which of the boundary
    // function values are bigger and use that as the boundary value for both f(a) and f(b)
    // when initializing the minimizer.
    // This circumvents the check done at initialization of the minimizer which would faultly state
    // that there is no minimum between a and b if our guess f(x) would happen to be bigger than a
    // or b.
    double f_max = llh.function(rPoint, &lh);
    double f_lower = llh.function(lPoint, &lh); 
    
    if(f_max!=f_max){
        sprintf(error_str,"Likelihood evaluation returns NaN\n error caught at at line %d in file %s",__LINE__,__FILE__);
        throw std::runtime_error(std::string(error_str));
    }

    if(std::isinf(f_max)){
        sprintf(error_str,"Likelihood evaluation returns inf\n error caught at at line %d in file %s",__LINE__,__FILE__);
        throw std::runtime_error(std::string(error_str));
    }

    if(f_max < f_lower)
        f_max = f_lower;
    
    status = gsl_min_fminimizer_set_with_values(ms_, &llh,
                                       mPoint, llh.function(mPoint, &lh),
                                       lPoint, f_max,
                                       rPoint, f_max);
    if(status != 0 && status != GSL_CONTINUE){
            std::string error(gsl_strerror (status));
            sprintf(error_str,"\n error caught at at line %d in file %s with error code %d",__LINE__,__FILE__,status);
            std::string error_continued(error_str);
            throw std::runtime_error(error+error_continued);
    }

    do{
        nIterations_++;
        status = gsl_min_fminimizer_iterate (ms_);
        
        if(status != 0 && status != GSL_CONTINUE){
            std::string error(gsl_strerror (status));
            sprintf(error_str,"\n error caught at at line %d in file %s with error code %d",__LINE__,__FILE__,status);
            std::string error_continued(error_str);
            throw std::runtime_error(error+error_continued);
        
        }
        
        lPoint = gsl_min_fminimizer_x_lower(ms_);
        rPoint = gsl_min_fminimizer_x_upper(ms_);

        status = gsl_min_test_interval(lPoint, rPoint, 1e-5, 0.0);

    }while (status == GSL_CONTINUE && nIterations_ < max_iter);

    if(status!=0){
        std::string error(gsl_strerror (status));
        sprintf(error_str,"error code %d, at line %d in file %s",status,__LINE__,__FILE__);
        std::string error_continued(error_str);
        throw std::runtime_error(error+error_continued);
    }

    bestFit_ = gsl_min_fminimizer_x_minimum(ms_);

    //evaluating the likelihood at the best fit point.
    bestFitLLH_ = lh.EvaluateLLH(bestFit_);

    return bestFitLLH_;

}
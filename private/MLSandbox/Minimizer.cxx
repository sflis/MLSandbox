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

double Minimizer::ComputeBestFit(Likelihood &lh){
    /// Determining the best fit (maximizing the likelihood) with the gsl library minimizer 
    /// using brent's method.
    lh.MinimizerConditions(*this);
    double searchInterval = 5e-3;
    double lPoint = 0; //Left starting point of the search interval method
    double rPoint = 1e-7;//Right starting point of the search interval
    double interval = rPoint - lPoint;
    double mPoint = interval/2 + lPoint; //Middle slope
    double lllh = lh.EvaluateLLH(lPoint);
    double rllh = lh.EvaluateLLH(rPoint);
    char error_str[300]; 
    double max = 0;
    double mllh =0;

    minXi_ = minMinimizerXi_;
    maxXi_ = maxMinimizerXi_;
    //Finding the general intervall in which 
    //the minimizer should work in
    if(lllh<rllh){//max of llh is at xi>0
        mllh = rllh;
        mPoint =rPoint;
    }
    else{//max of llh is at xi<0
        searchInterval *= -0.2;
        mllh = lllh;
        mPoint = lPoint;
    }
    int i =0;
    
    do{
        max = mllh;
        mPoint+=searchInterval;
        mllh = lh.EvaluateLLH(mPoint);
        i++;  
    }while(mllh>max && mPoint>minXi_ && mPoint<maxXi_);
    

    
    if(lllh>rllh){
        lPoint = mPoint;
        mPoint = mPoint - searchInterval;
        rPoint = mPoint - searchInterval;
    }
    else{
        rPoint = mPoint;
        mPoint = mPoint - searchInterval;
        lPoint = mPoint - searchInterval;   
    }

    if(lPoint<minXi_){
        lPoint=minXi_;
        if(mPoint<=lPoint)
            mPoint = lPoint+1e-8;

    }
    if(rPoint>maxXi_){
        rPoint=maxXi_;
        if(mPoint>=rPoint)
            mPoint = rPoint-1e-8;
    }
    
    //The left boundary can be undefined and thus
    //needs to be moved closer to the mid point
    lllh = lh.EvaluateLLH(lPoint);
    if(isnan(lllh)){
        double nmPoint = (mPoint-lPoint)/2 + lPoint;
        double nllh = lh.EvaluateLLH(nmPoint);
        double nlPoint = lPoint;
        
        while(nmPoint-nlPoint>1e-7 or isnan(nllh)){
            if(isnan(nllh)){
                nlPoint = nmPoint;
                nmPoint = (mPoint-nlPoint)/2 + nlPoint;
                nllh = lh.EvaluateLLH(nmPoint);
            }
            else{   
                nlPoint = (nmPoint-nlPoint)/2 + nlPoint;
                nllh = lh.EvaluateLLH(nlPoint);
            }
            
        }
        //f(mPoint) might be larger than
        //f(nlPooint)
        if(nllh>mllh)
            mPoint = nmPoint;

        lPoint = nlPoint;
        
    }

    

    // Variables needed for the gsl minimizer
    int status;
    int  max_iter = 30;
    nIterations_ = 0;
    gsl_function llh;

    llh.function = lh.CallBackFcn();
    llh.params = &lh;

    //switching to -llh
    rllh = llh.function(rPoint, &lh);
    lllh = llh.function(lPoint, &lh); 
    mllh = llh.function(mPoint, &lh);
    

    //Testing if min is at the boundary
    //or close to boundary
    double bllh = llh.function(lPoint+1e-8,&lh);
    if(lllh<bllh && lllh<mllh){
        bestFit_ = lPoint;
        bestFitLLH_ = lh.EvaluateLLH(bestFit_);
        return bestFitLLH_;
    }
    else if(lllh<mllh && lllh>bllh){
        mllh  =bllh;
        mPoint = lPoint+1e-8;
    }

    if(mPoint>lh.MaxXiBound() && lh.MaxXiBound()>0){
        mPoint = lh.MaxXiBound()-1e-14;
        mllh = llh.function(mPoint, &lh);
        if(mPoint<lPoint){
            lPoint = minXi_;
            lllh = llh.function(lPoint, &lh);
        }   
    }       

    if(rllh!=rllh){
        sprintf(error_str,"Likelihood evaluation returns NaN\n error caught at at line %d in file %s",__LINE__,__FILE__);
        throw std::runtime_error(std::string(error_str));
    }

    if(std::isinf(rllh)){
        sprintf(error_str,"Likelihood evaluation returns inf\n error caught at at line %d in file %s",__LINE__,__FILE__);
        throw std::runtime_error(std::string(error_str));
    }
    
    
    if(mllh!=mllh){
        sprintf(error_str,"Likelihood evaluation returns NaN\n error caught at at line %d in file %s",__LINE__,__FILE__);
        throw std::runtime_error(std::string(error_str));
    }

    if(std::isinf(mllh)){
        sprintf(error_str,"Likelihood evaluation returns inf\n error caught at at line %d in file %s",__LINE__,__FILE__);
        throw std::runtime_error(std::string(error_str));
    }    
    
    status = gsl_min_fminimizer_set_with_values(ms_, &llh,
                                       mPoint, mllh,
                                       lPoint, lllh,
                                       rPoint, rllh);
    
    if(status != 0 && status != GSL_CONTINUE){
            
            std::string error(gsl_strerror (status));
            sprintf(error_str,"\n error caught at at line %d in file %s with error code %d",__LINE__,__FILE__,status);
            std::string error_continued(error_str);
            std::cout<<lPoint<<" "<<mPoint<<" "<<rPoint<<std::endl;
            std::cout<<lllh-mllh<<" "<<mllh<<" "<<rllh-mllh<<std::endl;
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

        status = gsl_min_test_interval(lPoint, rPoint, 1e-7, 0.0);

    }while (status == GSL_CONTINUE && nIterations_ < max_iter );

    if(status!=0){
        std::string error(gsl_strerror (status));
        sprintf(error_str,"error code %d, at line %d in file %s",status,__LINE__,__FILE__);
        std::string error_continued(error_str);
        throw std::runtime_error(error+error_continued);
    }

    bestFit_ = gsl_min_fminimizer_x_minimum(ms_);
    //if(bestFit_ < minXi_)
    //    bestFit_=minXi_;
    //evaluating the likelihood at the best fit point.
    bestFitLLH_ = lh.EvaluateLLH(bestFit_);

    return bestFitLLH_;

}
//  Copyright (C) 2015
//  Samuel Flis <samuel.flis@fysik.su.se>
//  and the IceCube Collaboration <http://www.icecube.wisc.edu>
//
//  This file is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 3 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program.  If not, see <http://www.gnu.org/licenses/>


#ifndef MLSANDBOX_NEYMAN_ANALYSIS_H
#define MLSANDBOX_NEYMAN_ANALYSIS_H

#include "Likelihood.h"
#include "Minimizer.h"
#include "FCRanks.h"

#include <boost/shared_ptr.hpp>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include <inttypes.h>
#include <vector>
#include <string>
#include <math.h>
#include <iostream>

/**class: NeymanAnalysis is a class which encapsulates the
*         Neyman analysis proceedures.
**/

class  NeymanAnalysis{
    public:

        NeymanAnalysis(
            boost::shared_ptr<Likelihood> llh
        )
          : llh_(llh)
          , computedBestFit_(false)
          , ranksCompSet_(false)
          {
          }
        
        NeymanAnalysis(NeymanAnalysis &analysis, int64_t seed);

        ///Returns the log value of the FC test-statistic for a given likelihood parameter
        ///\param xi likelihood parameter for which the test statistic should be calculated
        ///\return the log of the FC test-statistic = log(L(xi)/L(xi_best))
        double EvaluateTestStatistic(double xi){
            if(!computedBestFit_){
                minimizer_.ComputeBestFit(*llh_);
                computedBestFit_ = true;
            }
            return minimizer_.bestFitLLH_ - (*llh_).EvaluateLLH(0);
        }


        std::vector<double> TestStatisticDistribution(double xi, uint64_t n);
        void ComputeRanks(uint64_t nExperiments,
                double minXi,
                double maxXi,
                uint64_t nSteps,
                uint64_t nThreads,
                uint64_t maxExperimentsPerThread);


        void SetFCRanks(FCRanks const &tsDistributions){
            ranksCompSet_ = true;
            tsDistributions_ = tsDistributions;
        }

        void Sample(double xi){
            llh_->SampleEvents(xi);
            computedBestFit_ = false;
        }

        //void GenerateTSEnsamble(double xiMin, double xiMax);

        std::vector<double> TestStatisticDistribution(double xi);

        double ComputeLimit(double ts, double cl, double prec);

        Minimizer minimizer_;
        FCRanks tsDistributions_;
private:

        boost::shared_ptr<Likelihood> llh_;

        bool computedBestFit_;
        bool assumeChi2_;
        bool ranksCompSet_;
};


#endif

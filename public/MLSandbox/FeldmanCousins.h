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


#ifndef MLSANDBOX_FELDMAN_COUSINS_H
#define MLSANDBOX_FELDMAN_COUSINS_H

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
#include <boost/numpy.hpp>
/**class: FeldmanCousinsAnalysis is a class which encapsulates the
*         FC analysis proceedures.
**/
namespace bn=boost::numpy;

class  FeldmanCousinsAnalysis{
    public:

        FeldmanCousinsAnalysis(
            boost::shared_ptr<Likelihood> llh,
            double cl
        )
          : llh_(llh)
          , cl_(cl)
          , computedBestFit_(false)
          , ranksCompSet_(false),
          likelihoodState_(0)
          {
          }

        ///Returns the log value of the FC test-statistic for a given likelihood parameter
        ///\param xi likelihood parameter for which the test statistic should be calculated
        ///\return the log of the FC test-statistic = log(L(xi)/L(xi_best))
        double EvaluateTestsStatistic(double xi){
            if(!computedBestFit_ || llh_->Changed()){
                minimizer_.ComputeBestFit(*llh_);
                computedBestFit_ = true;
            }
            return (*llh_).EvaluateLLH(xi) - minimizer_.bestFitLLH_;
        }

        void SetFCRanks(FCRanks const &ranks){
            ranksCompSet_ = true;
            ranks_ = ranks;
        }

        void Sample(double xi){
            llh_->SampleEvents(xi);
            computedBestFit_ = false;
        }

        ///Generates an ensemble of pseudo experiments to construct upper and
        ///lower limits distributions at a given likelihood parameter value xi
        ///\param xi
        ///\param up
        ///\param down
        ///\param cl
        ///\param nExperiments
        void GenerateLimitsEnsemble(double xi,
                                    bn::ndarray &up,
                                    bn::ndarray &down,
                                    uint64_t nExperiments,
                                    double cl = -1
                                    );

        void ComputeLimits(double &upper, double &lower);

        void ComputeLimits(double &upper, double &lower, bool assumeChi2){
            bool assumeChi2_old = assumeChi2_;
            ranks_.AssumeChiSqure(assumeChi2_);
            ComputeLimits(upper, lower);
            ranks_.AssumeChiSqure(assumeChi2_old);
        }

        void ComputeRanks(uint64_t nExperiments,
                double minXi,
                double maxXi,
                uint64_t nSteps,
                uint64_t nThreads = 1);

        FCRanks &GetRanks(){return ranks_;}
        FCRanks ranks_;
        FCRanks globalBestFits_;
        Minimizer minimizer_;
private:

        boost::shared_ptr<Likelihood> llh_;
        static double likelihoodRatioCrossings(double xi, void *params);

        double cl_;
        bool computedBestFit_;
        bool assumeChi2_;
        bool ranksCompSet_;
        uint32_t likelihoodState_;
};




#endif

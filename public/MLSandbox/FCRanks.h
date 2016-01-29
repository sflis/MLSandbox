//  Copyright (C) 2012
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


#ifndef MLSANDBOX_FCRANKS_H
#define MLSANDBOX_FCRANKS_H

#include "Likelihood.h"
#include "Minimizer.h"
#include <boost/shared_ptr.hpp>

// #include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/version.hpp>


#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

#include <inttypes.h>
#include <vector>
#include <map>
#include <string>
#include <math.h>
#include <iostream>
#include <algorithm>
const unsigned int FCRANKS_VERSION = 1;
/**class: FCRanks
 * \brief A class that holds Feldman Cousins rank distribtions and provides
 *        some utilities to interpolate the critical boundary at different
 *        confidence levels.
**/
class FCRanks{
    public:

        FCRanks():assumeChiSquareDistr_(true){
            acc_ = gsl_interp_accel_alloc ();
            spline_ = gsl_spline_alloc (gsl_interp_cspline, 10);
        }

        FCRanks(const FCRanks & base)
            :ranks_(base.ranks_)
            ,cl_(base.cl_)
            ,lastRCritValue_(base.lastRCritValue_)
            ,assumeChiSquareDistr_(base.assumeChiSquareDistr_)
            {
            acc_ = gsl_interp_accel_alloc ();
            spline_ = gsl_spline_alloc (gsl_interp_cspline, 10);

        }


        FCRanks & operator=(const FCRanks &rh){
            if(this != &rh){
                ranks_ = rh.ranks_;
                cl_ = rh.cl_;
                lastRCritValue_ = rh.lastRCritValue_;
                assumeChiSquareDistr_ = rh.assumeChiSquareDistr_;

            }
            return *this;
        }

        ~FCRanks(){
            gsl_spline_free (spline_);
            gsl_interp_accel_free (acc_);
        }

        ///Adds a rank distribution to the collection for a specific likelihood hypothesis
        ///\param xi the likelihood parameter value
        ///\param ranks a vector with ranks
        ///\param set if true the critical boundary is interpolated at the preset confidence
        ///           level cl_
        void Fill(double xi, std::vector<double> &ranks, bool set = true){
            std::sort(ranks.begin(),ranks.end());
            ranks_[xi] = ranks;
            if(set)
                SetConfidenceLevel(cl_);
        }

        void AssumeChiSqure(bool assume){
            if(ranks_.empty()){
                assumeChiSquareDistr_ = true;
            }
            else{
                assumeChiSquareDistr_ = assume;
            }
        }

        void SetConfidenceLevel(double cl){
            if(ranks_.size()<3){
                cl_ = cl;
                assumeChiSquareDistr_ = true;
            }
            assumeChiSquareDistr_ = false;
            gsl_spline_free (spline_);
            spline_ = gsl_spline_alloc (gsl_interp_cspline, ranks_.size());

            double x[ranks_.size()];
            double y[ranks_.size()];
            uint32_t index = 0;

            for (std::map<double, std::vector<double> >::iterator it=ranks_.begin(); it!=ranks_.end(); ++it,++index){
                    x[index] = it->first;
                    uint32_t cl_index = (it->second.size()-1)*(1-cl);
                    y[index] = it->second[cl_index];
            }

            lastRCritValue_ = x[index-2];
            Smooth(x, y, ranks_.size());
            gsl_spline_init (spline_, x, y, ranks_.size());
        }
        ///Returns the ln(Rank) at the preset confidence level
        ///\param xi
        double rCB(double xi, bool assumeChiSquareDistr=false){
            // If it is set that ln(R) follows a Chi-square distribution only this number is returned.
            if(assumeChiSquareDistr or assumeChiSquareDistr_)
                return -gsl_cdf_chisq_Pinv(cl_, 1)/2.0;

            // If xi is greater than the upper limit of the spline interpolation it is assumed constant
            // with the value at this limit.
            if(xi > lastRCritValue_)
                xi = lastRCritValue_;

            return gsl_spline_eval (spline_, xi, acc_);
        }
        std::map<double, std::vector<double> > ranks_;
    private:

        void Smooth(double x[],double y[],uint64_t n){
            //First smooth pass
            for(uint64_t i = 1; i < n-1; i++){
                if(i > 2 && i < n-3){
                    y[i] = (y[i-3] + y[i-2] + y[i-1] + y[i] + y[i+1] + y[i+2] + y[i+3])/7.0;
                }
                else{
                    y[i] = (y[i-1] + y[i] + y[i+1])/3.0;
                }
            }

            y[n-1] = (y[n-3] + y[n-2] + y[n-1])/3;
            //Second smooth pass
            for(uint64_t i = 1; i < n-1; i++){
                if(i > 2 && i < n-3){
                    y[i] = (y[i-3] + y[i-2] + y[i-1] + y[i] + y[i+1] + y[i+2] + y[i+3])/7.0;
                }
                else{
                    y[i] = (y[i-1] + y[i] + y[i+1])/3.0;
                }
            }

            y[n-1] = (y[n-3] + y[n-2] + y[n-1])/3;


        }



        double cl_;
        gsl_interp_accel *acc_;
        gsl_spline *spline_;
        double lastRCritValue_;
        bool assumeChiSquareDistr_;
    public:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & ranks_;

        }
};

namespace mlsandbox{
    void saveFCRanks(std::string file_name, FCRanks &ranks);

    FCRanks loadFCRanks(std::string file_name);

}

BOOST_CLASS_VERSION(FCRanks, FCRANKS_VERSION )

#endif

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

#ifndef MLSANDBOX_COMBINED_LIKELIHOOD_H
#define MLSANDBOX_COMBINED_LIKELIHOOD_H

#include "Likelihood.h"
#include <vector>

class CombinedLikelihood: public Likelihood{
    public:
        CombinedLikelihood(const std::vector<boost::shared_ptr<Likelihood> > & likelihoods_,
                           const std::vector<double> & weights_
            ):
        likelihoods_(likelihoods_),
        weights_(weights_)
        {
            totEvents_ = 0;
            hashes_.resize(likelihoods_.size());
            N_ = 0;
            for(uint64_t i = 0; i<likelihoods_.size(); i++){
                totEvents_ += likelihoods_[i]->totEvents_;
                hashes_[i] = likelihoods_[i]->StateHash();
                N_ += likelihoods_[i]->N_;
            }
            changed_ = true;
        }

        ///Evaluates the log likelihood sum
        ///\param xi the signal fraction for which the likelihood should be evaulated.
        double EvaluateLLH(double xi)const;

        void SampleEvents(double xi);

        likelihoodCallback CallBackFcn(){return &likelihoodEval;}

        CombinedLikelihood * Clone(int seed)const{
            std::vector<boost::shared_ptr<Likelihood> > llhv;
            for(uint64_t i = 0; i<likelihoods_.size(); i++){
                llhv.push_back(boost::shared_ptr<Likelihood>(likelihoods_[i]->Clone(seed)));
            }

            return new CombinedLikelihood(llhv,weights_);
        }

        uint32_t StateHash(){
            for(uint64_t i = 0; i<likelihoods_.size(); i++){
                hashes_[i] = likelihoods_[i]->StateHash();
            }
            return SuperFastHash((const char*) &hashes_[0],sizeof(uint32_t)*hashes_.size());
        }

        double MaxXiBound();
        void Update(){
            totEvents_ = 0;
            for(uint64_t i = 0; i<likelihoods_.size(); i++){
                totEvents_ += likelihoods_[i]->totEvents_;
            }
            changed_ = true;
        }
    private:
        static double likelihoodEval(double xi, void *params);
        std::vector<boost::shared_ptr<Likelihood> > likelihoods_;
        std::vector<double> weights_;
        std::vector<int32_t> hashes_;

};


#endif

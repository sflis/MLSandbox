#ifndef FELDMAN_COUSINS_LIKELIHOOD_COMBINED_H
#define FELDMAN_COUSINS_LIKELIHOOD_COMBINED_H
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
                hashes_[i] = likelihoods_[i]->ChangedHash();
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

        uint32_t ChangedHash(){
            for(uint64_t i = 0; i<likelihoods_.size(); i++){
                hashes_[i] = likelihoods_[i]->ChangedHash();
            }
            return SuperFastHash((const char*) &hashes_[0],sizeof(uint32_t)*hashes_.size());
        }
    private:
        static double likelihoodEval(double xi, void *params);
        std::vector<boost::shared_ptr<Likelihood> > likelihoods_;
        std::vector<double> weights_;
        std::vector<int32_t> hashes_;

};


#endif

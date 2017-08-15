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
#ifndef BINNED_DISTRIBUTION_H
#define BINNED_DISTRIBUTION_H

#include <inttypes.h>
#include <vector>
#include <boost/shared_ptr.hpp>

#include "RNG.h"
#include <algorithm>

/**class: Distribution
*\brief A class that provides a common interface for binned distributions which are used
*       to represent binned pdfs in likelihood objects.
*
*
*/
class Distribution{
   public:

        Distribution(Distribution const &base, boost::shared_ptr<RNG> rng);
        Distribution(std::vector<double>const &distribution, double rMin, double rMax, unsigned int rSeed = 1);

        /// Returns the pdf value evaluated at the given value
        double PDF(double x) const; 

        /// Fast unsafe call to PDF (no boundary checks)
        /// \param x the pdf paramater at which the pdf should be evaluated at
        /// \return pdf value at x
        double PDF_f(double x) const{
            return pdf_[(x - rangeMin_) * invBinWidth_];
        }

        /// Evaluates the CDF at a given value
        double CDF(double x) const;

        /// Returns a random number sampled according to the binned distribution
        double SampleFromDistr() const;

        /// Returns a random index of the binned distribution sampled according to the same distribution
        uint64_t SampleFromDistrI()const{
           return  std::upper_bound(cdf_.begin(), cdf_.end(), rng_->Uniform()) - cdf_.begin();
        }


        /// Returns the number of bins of the pdf
        uint64_t GetNBins()const {return nBins_;}

        /// Returns the bin index of the binned pdf corresponding to the givem value
        uint64_t ValueToBin(double x){return (x - rangeMin_) * invBinWidth_ + 1;}

        /// Returns a reference to the pdf
        std::vector<double> & GetPDFVector(){return pdf_;}
        
        
        /// Provides direct access to the binned pdf 
        double operator[](uint64_t index){return pdf_[index];}
        
        /// Provides direct access to the binned pdf (const)
        double operator[] (uint64_t index) const {return pdf_[index];}
        
        void SetCDFSampling(bool set){useCDF_ = set;}

        double GetRangeMax() const{return rangeMax_;}
        
        double GetRangeMin() const{return rangeMin_;}
   private:
        friend void addDistributions(double w1,
                                     Distribution const &dst1,
                                     double w2,
                                     Distribution const &dst2,
                                     Distribution & target);
        double rangeMax_;
        double rangeMin_;
        bool useCDF_;
        double pdfMax_;
        uint32_t seed_;
        std::vector<double> pdf_;
        std::vector<double> cdf_;
        double binWidth_;
        double invBinWidth_;
        double range_;
        uint64_t nBins_;
        boost::shared_ptr<RNG> rng_;

};


///A helper function that performs an addition operation on two 
///distributions and putting the result in a third distribution object  
inline void addDistributions(double w1, 
                             Distribution const &dst1, 
                             double w2, 
                             Distribution const &dst2, 
                             Distribution & target){
        
        const uint64_t n = dst1.nBins_;
        target.pdf_[0] = w1*dst1.pdf_[0] + w2*dst2.pdf_[0];
        target.cdf_[0] = target.pdf_[0];

        for(uint64_t i = 1; i<n; i++){
            target.pdf_[i] = w1*dst1.pdf_[i] + w2*dst2.pdf_[i];
            target.cdf_[i] = target.cdf_[i-1] + target.pdf_[i];
        }
        //Finding the max value of the new pdf
        target.pdfMax_ = *std::max_element(target.pdf_.begin(),target.pdf_.end());
}


#endif

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

typedef double(*ptDFctD)(double);
/**class: Distribution
*\brief A class that provides a common interface for distributions which can be used to build up
* likelihoods.
*
*
*/
class Distribution{
   public:

        Distribution(Distribution const &base, boost::shared_ptr<RNG> rng);
        Distribution(ptDFctD pdf, double rMin, double rMax, uint64_t nBins, uint rSeed = 1);
        Distribution(std::vector<double>const &distribution, double rMin, double rMax, uint rSeed = 1);

        double PDF(double x) const;

        /// Fast unsafe call to PDF (no boundary checks)
        /// \param x the pdf paramater at which the pdf should be evaluated at
        /// \return pdf value at x
        double PDF_f(double x) const{
            return pdf_[(x - rangeMin) * invBinWidth_];
        }

        double CDF(double x) const;

        double SampleFromDistr() const;

        double SampleFromDistrI() const;

        uint64_t GetNBins()const {return nBins_;}
        uint64_t ValueToBin(double x){return (x - rangeMin) * invBinWidth_ + 1;}

        std::vector<double> & GetPDFVector(){return pdf_;}
        double operator[](uint64_t index){return pdf_[index];}
        double operator[] (uint64_t index) const {return pdf_[index];}
        void SetCDFSampling(bool set){useCDF_ = set;}

        double GetRangeMax() const{return rangeMax;}
        double GetRangeMin() const{return rangeMin;}
   private:
        friend void addDistributions(double w1,Distribution const &dst1,double w2,Distribution const &dst2,Distribution & target);
        double rangeMax;
        double rangeMin;
        bool cdf;
        bool pdf;
        bool useCDF_;
        double pdfMax;
        uint32_t seed;
        std::vector<double> pdf_;
        std::vector<double> cdf_;
        double binWidth_;
        double invBinWidth_;
        double range_;
        uint64_t nBins_;
        boost::shared_ptr<RNG> rng;

};



inline void addDistributions(double w1, Distribution const &dst1, double w2, Distribution const &dst2, Distribution & target){
        uint64_t n = dst1.nBins_;
        target.pdf_[0] = w1*dst1.pdf_[0] + w2*dst2.pdf_[0];
        target.cdf_[0] = target.pdf_[0];
        target.pdfMax = 0;
        for(uint64_t i = 1; i<n; i++){
            target.pdf_[i] = w1*dst1.pdf_[i] + w2*dst2.pdf_[i];
            target.cdf_[i] = target.cdf_[i-1] + target.pdf_[i];
            target.pdfMax = target.pdf_[i]>target.pdfMax ? target.pdf_[i]: target.pdfMax;
        }

}


#endif

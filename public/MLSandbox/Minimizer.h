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
#ifndef MLSANDBOX_MINIMIZER_H
#define MLSANDBOX_MINIMIZER_H

#include <inttypes.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_min.h>


#include "Likelihood.h"
/**class: Minimizer
*\brief A small wrapper class around the GSL minimizer
*
*/
class Minimizer{
        
public:
    
    Minimizer():nIterations_(0){
        const gsl_min_fminimizer_type *T = gsl_min_fminimizer_brent;
        ms_ = gsl_min_fminimizer_alloc (T);
    }
    
    ~Minimizer(){gsl_min_fminimizer_free (ms_);}
    
    ///Computes the best fit given a Likelihood
    ///\param lh a likelihood object
    ///\return best fit 
    double ComputeBestFit(Likelihood &lh);
    ///Best fit of the likelihood parameter from the last fit
    double bestFit_;
    ///Log of the likelihood value at the best fit 
    double bestFitLLH_;
    
    uint64_t nIterations_;
    
private:
    gsl_min_fminimizer *ms_;

};

#endif

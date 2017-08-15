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
#ifndef RNG_MLSANDBOX_H
#define RNG_MLSANDBOX_H

#include <inttypes.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

/**class: RNG
*\brief A small wrapper class around the GSL random number generator
*
*/
class RNG{
   public:
        RNG(unsigned int seed):seed_(seed){
            const gsl_rng_type *T = gsl_rng_default;
            rng_ = gsl_rng_alloc(T);
            gsl_rng_set(rng_, seed);
        }
        ~RNG(){gsl_rng_free (rng_);}

        double Uniform(){return gsl_rng_uniform(rng_);}
        double Exp(double tau){return gsl_ran_exponential(rng_, tau);}
        double Poisson(double nu){return gsl_ran_poisson(rng_, nu);}
        double Gauss(double mu, double sigma){return gsl_ran_gaussian(rng_, sigma) + mu;}
        double Binomial(double p, double N){ return gsl_ran_binomial(rng_, p,N);}
        unsigned int seed_;
    private:
        gsl_rng *rng_;

};

#endif

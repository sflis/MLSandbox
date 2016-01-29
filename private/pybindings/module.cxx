/* Copyright (C) 2012
 * Samuel Flis <samuel.d.flis@gmail.com>
 * and the IceCube Collaboration <http://www.icecube.wisc.edu>
 *
 * This file is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */
#include <boost/preprocessor.hpp>

#include <boost/numpy.hpp>
using namespace boost::python;

#define REGISTER_THESE_THINGS \
    (Distribution)\
    (Likelihood)\
    (Minimizer)\
    (FCRanks)\
    (FeldmanCousins)\
    (NeymanAnalysis)

#define I3_REGISTRATION_FN_DECL(r, data, t) void BOOST_PP_CAT(register_,t)();
#define I3_REGISTER(r, data, t) BOOST_PP_CAT(register_,t)();

BOOST_PP_SEQ_FOR_EACH(I3_REGISTRATION_FN_DECL, ~, REGISTER_THESE_THINGS)

BOOST_PYTHON_MODULE(MLSandbox)
{
    boost::numpy::initialize();
    BOOST_PP_SEQ_FOR_EACH(I3_REGISTER, ~, REGISTER_THESE_THINGS);
}

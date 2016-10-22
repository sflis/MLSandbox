#include "MLSandbox/NeymanAnalysis.h"
// #include "bindingutils.h"

#include <boost/numpy.hpp>

#include <boost/python/tuple.hpp>
#include <iostream>
#include <vector>

using namespace std;
namespace bp=boost::python;
namespace bn=boost::numpy;

namespace mlsandbox{
    /*namespace python{

        static bp::tuple GenerateLimitsEnsemble(FeldmanCousinsAnalysis &self, double xi,double cl, uint64_t nExperiments){
            std::vector<double> up;
            std::vector<double> down;
            self.GenerateLimitsEnsemble(xi, up, down, cl, nExperiments);
            return bp::make_tuple(0.0,0.0);
        }

       static bp::tuple ComputeLimits(FeldmanCousinsAnalysis &self){
            double up,down;
            self.ComputeLimits(up,down);
            return bp::make_tuple(up,down);
       }
    }
*/
}


//using namespace boost::python;
namespace bp=boost::python;
//using namespace mlsandbox::python;
void register_NeymanAnalysis()
{

    bp::scope NeymanAnalysis_scope =
    bp::class_<NeymanAnalysis, boost::shared_ptr<NeymanAnalysis> >
        ("NeymanAnalysis","DocString",
            bp::init<boost::shared_ptr<Likelihood> >(( bp::args("llh"))
                    ,"Constructor for FeldmanCousins analysis"
                    )
        )
       .def("EvaluateTestStatistic",&NeymanAnalysis::EvaluateTestStatistic,("xi"))
       .def("Sample",&NeymanAnalysis::Sample,("xi"))
       .def("SetFCRanks",&NeymanAnalysis::SetFCRanks)
       .def("ComputeRanks",&NeymanAnalysis::ComputeRanks,(
            bp::args("n_experiments")
            ,bp::args("min_xi")
            ,bp::args("max_xi")
            ,bp::args("n_steps")
            ,bp::args("n_threads")
            ,bp::args("max_n_experiments")
            )
        )
       .def_readwrite("ranks",&NeymanAnalysis::tsDistributions_)
       //.def("GenerateTSEnsamble",&NeymanAnalysis::GenerateTSEnsamble)
       .def("ComputeLimit",&NeymanAnalysis::ComputeLimit)
       .def_readwrite("minimizer",&NeymanAnalysis::minimizer_)
       ;

}

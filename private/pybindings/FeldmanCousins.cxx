#include "MLSandbox/FeldmanCousins.h"
// #include "bindingutils.h"

#include <boost/numpy.hpp>

#include <boost/python/tuple.hpp>
#include <iostream>
#include <vector>

using namespace std;
namespace bp=boost::python;
namespace bn=boost::numpy;

namespace mlsandbox{
    namespace python{

        static bp::tuple GenerateLimitsEnsemble(FeldmanCousinsAnalysis &self, double xi,double cl, uint64_t nExperiments){
            std::vector<intptr_t> shape(1,nExperiments);
            bn::dtype dt = bn::dtype::get_builtin<double>();
            bn::ndarray up = bn::zeros(shape,dt);
            bn::ndarray down = bn::zeros(shape,dt);
            self.GenerateLimitsEnsemble(xi, up, down, nExperiments, cl);
            return bp::make_tuple(up,down);
        }

       static bp::tuple ComputeLimits(FeldmanCousinsAnalysis &self){
            double up,down;
            self.ComputeLimits(up,down);
            return bp::make_tuple(up,down);
       }
    }

}


//using namespace boost::python;
namespace bp=boost::python;
using namespace mlsandbox::python;
void register_FeldmanCousins()
{
  {
    bp::scope distribution_scope =
    bp::class_<FeldmanCousinsAnalysis, boost::shared_ptr<FeldmanCousinsAnalysis> >
      ("FeldmanCousinsAnalysis","DocString",
           bp::init<boost::shared_ptr<Likelihood>, double>(
                        (
                            bp::args("llh")
                            ,bp::args("cl")
                        ),
                    "Constructor for FeldmanCousins analysis"
                    )
    )
       .def("EvaluateTestsStatistic",&FeldmanCousinsAnalysis::EvaluateTestsStatistic,("xi"))
       .def("Sample",&FeldmanCousinsAnalysis::Sample,("xi"))
       .def("ComputeRanks",&FeldmanCousinsAnalysis::ComputeRanks,(
            bp::args("n_experiments")
            ,bp::args("min_xi")
            ,bp::args("max_xi")
            ,bp::args("n_steps")
            ,bp::args("n_threads")
            )
        )
       .def("SetFCRanks",&FeldmanCousinsAnalysis::SetFCRanks)
       .def("ComputeLimits",&mlsandbox::python::ComputeLimits)
       .def("GenerateLimitsEnsemble",&mlsandbox::python::GenerateLimitsEnsemble)
       .def_readwrite("ranks",&FeldmanCousinsAnalysis::ranks_)
       .def_readwrite("bestfits",&FeldmanCousinsAnalysis::globalBestFits_)
       .def_readwrite("minimizer",&FeldmanCousinsAnalysis::minimizer_)
       ;
  }
}

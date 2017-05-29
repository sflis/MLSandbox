#include "MLSandbox/Minimizer.h"
#include "bindingutils.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
//namespace mlsandbox{
//    namespace python{
//
//        }
//}


//using namespace boost::python;
namespace bp=boost::python;
//using namespace mlsandbox::python;
void register_Minimizer()
{
  {
    bp::scope distribution_scope =
    bp::class_<Minimizer, boost::shared_ptr<Minimizer> >
      ("Minimizer","DocString",
        bp::init<>(
                   "Constructor for MLSandbox Minimizer"
                   )
    )
        .def("ComputeBestFit",&Minimizer::ComputeBestFit)
        .def("SetBoundaries",&Minimizer::SetBoundaries)
        .def_readonly("bestFit",&Minimizer::bestFit_)
        .def_readonly("bestFitLLH",&Minimizer::bestFitLLH_)
        .def_readonly("nIterations",&Minimizer::nIterations_)
       ;
  }
}

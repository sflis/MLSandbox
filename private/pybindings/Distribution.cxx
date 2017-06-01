#include "MLSandbox/Distribution.h"
// #include "bindingutils.h"
#include <boost/numpy.hpp>

#include <iostream>
namespace bn = boost::numpy;
namespace bp = boost::python;
using namespace std;
namespace mlsandbox{
    namespace python{
        boost::shared_ptr<Distribution> distribtion_constructor(bp::object distribution,
                                                                     double rMin,
                                                                     double rMax,
                                                                     uint rSeed){
            bn::ndarray arr = bn::from_object(distribution,bn::dtype::get_builtin<double>(), 1, 1, bn::ndarray::CARRAY_RO);
            std::vector<double> distribution_vec(arr.get_size());
            std::copy((double*)arr.get_data(), (double*)arr.get_data()+arr.get_size(), &distribution_vec[0]);
            return boost::shared_ptr<Distribution>( new Distribution(distribution_vec, rMin, rMax, rSeed) );
        }

        bn::ndarray sample(Distribution &self, int N){
            std::vector<intptr_t> shape(1,N);
            bn::dtype dt = bn::dtype::get_builtin<double>();
            bn::ndarray sample_array = bn::empty(shape,dt);
            double* data = (double*) sample_array.get_data();
            for(uint i = 0; i<N; ++i){
                *data = self.SampleFromDistr();
                data++;
            }
            return sample_array;            
        }
    }
}

//using namespace boost::python;
namespace bp=boost::python;
using namespace mlsandbox::python;
void register_Distribution()
{
  {
    bp::scope distribution_scope =
    bp::class_<Distribution, boost::shared_ptr<Distribution> >
      ("Distribution","DocString",bp::no_init)
       .def("__init__", make_constructor(&distribtion_constructor), "__init__ docstring")
       .def("Sample",&Distribution::SampleFromDistr)
       .def("SampleI",&Distribution::SampleFromDistrI)
       .def("PDF",&Distribution::PDF)
       .def("CDF",&Distribution::CDF)
       .def("SetCDFSampling",&Distribution::SetCDFSampling)
       .def("SampleN",mlsandbox::python::sample)
       ;
  }
}

#include "MLSandbox/Likelihood.h"
#include "MLSandbox/Distribution.h"
#include "MLSandbox/CombinedLikelihood.h"
#include "bindingutils.h"
// #include "bindingutils.h"

#include <boost/numpy.hpp>
#include <numpy/arrayobject.h>
namespace bp = boost::python;
namespace bn = boost::numpy;

namespace mlsandbox{
    namespace python{
        void set_events(BinnedLikelihood &self,  bp::object obj){
            bn::ndarray arr = bn::from_object(obj,bn::dtype::get_builtin<double>(), 1, 1, bn::ndarray::CARRAY_RO);
            std::vector<double> vec(arr.get_size());
            std::copy((double*)arr.get_data(), (double*)arr.get_data()+arr.get_size(), &vec[0]);
            self.SetEvents(vec);
        }

        boost::shared_ptr<CombinedLikelihood> initWrapperCombinedLikelihood(bp::object const & l_list, bp::object const & w_list){
            bp::list likelihood_list  = bp::extract<bp::list>(l_list);
            bp::list weight_list  = bp::extract<bp::list>(w_list);
            std::vector<boost::shared_ptr<Likelihood> > likelihoods = toStdVector<boost::shared_ptr<Likelihood> >(likelihood_list);
            std::vector<double> weights = toStdVector<double>(weight_list);
            return boost::shared_ptr<CombinedLikelihood>(new CombinedLikelihood(likelihoods,weights));
        }
    }//name space PYTHON
}//name space MLSANDBOX

//using namespace boost::python;
namespace bp=boost::python;
using namespace mlsandbox::python;
void register_Likelihood()
{

    // Tell boost::python that it should create the python docstrings with
    // user-defined docstrings, python signatures, but no C++ signatures
    bp::docstring_options docstring_opts(true, true, false);

    // Construct a boost::python object, representing the Python None object,
    // which can than be used as default arguments in class method definitions.
    bp::object bp_Py_None(bp::handle<>(bp::borrowed(Py_None)));
    {
    ////////////////////////////////////////////////////////////////////////////
    // Define baseclass.
    //--------------------------------------------------------------------------
    // Likelihood
    bp::scope Likelihood_scope = bp::class_<Likelihood, boost::shared_ptr<Likelihood>, boost::noncopyable>("Likelihood",
        "Likelihood base class                                                    \n",
        bp::no_init)
    ;

    {
        bp::scope CombinedLikelihood_scope =
        bp::class_<CombinedLikelihood, boost::shared_ptr<CombinedLikelihood>,bp::bases<Likelihood> >
        ("CombinedLikelihood","DocString", bp::no_init
        //bp::init<const std::vector<boost::shared_ptr<Likelihood> > &,
        //const std::vector<double> & >
        //(
        //            "Constructor for a combined likelihood"
        //            )
        )
        .def("__init__",bp::make_constructor(&mlsandbox::python::initWrapperCombinedLikelihood)
//            ,(bp::args("likelihood_list"),
//            bp::args("weight_list")
//            )
        )
        .def("SampleEvents",&ShapeLikelihood::SampleEvents)
        .def("EvaluateLLH",&ShapeLikelihood::EvaluateLLH)
        ;

    }//CombinedLikelihood


    {
    ////////////////////////////////////////////////////////////////////////////
    // Define baseclass.
    //--------------------------------------------------------------------------
    // BinnedLikelihood
    bp::scope BinnedLikelihood_scope = bp::class_<BinnedLikelihood, boost::shared_ptr<BinnedLikelihood>, bp::bases<Likelihood>, boost::noncopyable>("BinnedLikelihood",
        "BinnedLikelihood base class                                                    \n",
        bp::no_init)
    .def("SetEvents",&mlsandbox::python::set_events)
    ;


    {
        bp::scope SignalContaminatedLH_scope =
        bp::class_<SignalContaminatedLH, boost::shared_ptr<SignalContaminatedLH>,bp::bases<BinnedLikelihood> >
        ("SignalContaminatedLH","DocString",
        bp::init<Distribution &,
                    Distribution &,
                    Distribution &,
                    Distribution &,
                    Distribution &,
                    Distribution &,
                    double,
                    double,
                    double,
                    SignalContaminatedLH::Model,
                    double,
                    double,
                    int>(
                    "Constructor for signal contaminated likelihood"
                    )
        )
        .def("SampleEvents",&SignalContaminatedLH::SampleEvents)
        .def("EvaluateLLH",&SignalContaminatedLH::EvaluateLLH)
        .def("EnableHistogramedEvents",&SignalContaminatedLH::EnableHistogramedEvents)
        ;

        bp::enum_<SignalContaminatedLH::Model>("Model")
        .value("None", SignalContaminatedLH::None)
        .value("Poisson", SignalContaminatedLH::Poisson)
        .value("Binomial", SignalContaminatedLH::Binomial)
        .export_values();
    }//SignalContaminatedLH scope

    {
        bp::scope ShapeLikelihood_scope =
        bp::class_<ShapeLikelihood, boost::shared_ptr<ShapeLikelihood>,bp::bases<BinnedLikelihood> >
        ("ShapeLikelihood","DocString",
        bp::init<Distribution &,
                    Distribution &,
                    Distribution &,
                    Distribution &,
                    double,
                    int>(
                    "Constructor for a simple shape likelihood"
                    )
        )
        .def("SampleEvents",&ShapeLikelihood::SampleEvents)
        .def("EvaluateLLH",&ShapeLikelihood::EvaluateLLH)
        .def("EnableHistogramedEvents",&ShapeLikelihood::EnableHistogramedEvents)
        .def("EnablePoissonSampling",&ShapeLikelihood::EnablePoissonSampling)
        ;

    }//ShapeLikelihood scope
    }//BinnedLikelihood scope

    }//Likelihood scope
}

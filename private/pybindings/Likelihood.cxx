#include "MLSandbox/Likelihood.h"
#include "MLSandbox/Distribution.h"
#include "MLSandbox/CombinedLikelihood.h"
#include "MLSandbox/SignalContaminatedLH.h"
#include "MLSandbox/PSignalContaminatedLH.h"
#include "MLSandbox/LikelihoodCollection.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "bindingutils.h"

#include <boost/python.hpp>


#include <boost/python/tuple.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/dict.hpp>

#include <boost/numpy.hpp>
#include <numpy/arrayobject.h>
namespace bp = boost::python;
namespace bn = boost::numpy;

namespace mlsandbox{
    namespace python{

        bn::ndarray get_events(BinnedLikelihood &self){
            std::vector<uint64_t> &sample = self.GetEventSample();
            std::vector<intptr_t> shape(1,sample.size());
            bn::dtype dt = bn::dtype::get_builtin<uint64_t>();
            bn::ndarray sample_array = bn::zeros(shape,dt);
            std::copy(&sample[0],&sample[0]+sample.size(),(uint64_t*)sample_array.get_data());
            return sample_array;

        }

        void set_events(BinnedLikelihood &self,  bp::object obj){
            bn::ndarray arr = bn::from_object(obj,bn::dtype::get_builtin<uint64_t>(), 1, 1, bn::ndarray::CARRAY_RO);
            std::vector<uint64_t> vec(arr.get_size());
            std::copy((uint64_t*)arr.get_data(), (uint64_t*)arr.get_data()+arr.get_size(), &vec[0]);
            self.SetEvents(vec);
        }

        boost::shared_ptr<CombinedLikelihood> initWrapperCombinedLikelihood(bp::object const & l_list, bp::object const & w_list){
            bp::list likelihood_list  = bp::extract<bp::list>(l_list);
            bp::list weight_list  = bp::extract<bp::list>(w_list);
            std::vector<boost::shared_ptr<Likelihood> > likelihoods = toStdVector<boost::shared_ptr<Likelihood> >(likelihood_list);
            std::vector<double> weights = toStdVector<double>(weight_list);
            return boost::shared_ptr<CombinedLikelihood>(new CombinedLikelihood(likelihoods,weights));
        }


        // void set_call(LikelihoodCollection &self,  bp::object obj){
        //    double (func)(const LikelihoodCollection &, double);
        //    func =  bp::extract<double ()(const LikelihoodCollection &, double)>(object)
            
        //     self.SetLLHFunction(func);
        // }

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
        .def_readwrite("N",&Likelihood::N_)
        .def_readonly("totEvents",&Likelihood::totEvents_)

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
        .def("SampleEvents",&CombinedLikelihood::SampleEvents)
        .def("EvaluateLLH",&CombinedLikelihood::EvaluateLLH)
        .def("Update",&CombinedLikelihood::Update)
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
    .def("GetEvents",&mlsandbox::python::get_events)
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
        //.def("EnableHistogramedEvents",&SignalContaminatedLH::EnableHistogramedEvents)
        ;

        bp::enum_<SignalContaminatedLH::Model>("Model")
        .value("None", SignalContaminatedLH::None)
        .value("Poisson", SignalContaminatedLH::Poisson)
        .value("Binomial", SignalContaminatedLH::Binomial)
        .export_values();
    }//SignalContaminatedLH scope


    {
        bp::scope LikelihoodCollection_scope =
        bp::class_<LikelihoodCollection, boost::shared_ptr<LikelihoodCollection>,bp::bases<BinnedLikelihood> >
        ("LikelihoodCollection","DocString",
        bp::init<Distribution &,
                    Distribution &,
                    Distribution &,
                    Distribution &,
                    Distribution &,
                    Distribution &,
                    double,
                    double,
                    double,
                    LikelihoodCollection::Model,
                    double,
                    double,
                    int>(
                    "Constructor for signal contaminated likelihood"
                    )
        )
        .def("SampleEvents",&LikelihoodCollection::SampleEvents)
        .def("EvaluateLLH",&LikelihoodCollection::EvaluateLLH)
        .def("standardSigSub",&LikelihoodCollection::standardSigSub)
        .staticmethod( "standardSigSub" )
        .def("noSigSubCorr",&LikelihoodCollection::noSigSubCorr)
        .staticmethod( "noSigSubCorr" )
        .def("SetLLHFunction",&LikelihoodCollection::SetLLHFunction)
        ;

        bp::enum_<LikelihoodCollection::Model>("Model")
        .value("None", LikelihoodCollection::None)
        .value("Poisson", LikelihoodCollection::Poisson)
        .value("Binomial", LikelihoodCollection::Binomial)
        .export_values();
    }//LikelihoodCollection


    {
        bp::scope PSignalContaminatedLH_scope =
        bp::class_<PSignalContaminatedLH, boost::shared_ptr<PSignalContaminatedLH>,bp::bases<BinnedLikelihood> >
        ("PSignalContaminatedLH","DocString",
        bp::init<Distribution &,
                    Distribution &,
                    Distribution &,
                    Distribution &,
                    Distribution &,
                    Distribution &,
                    double,
                    double,
                    double,
                    PSignalContaminatedLH::Model,
                    double,
                    double,
                    int>(
                    "Constructor for signal contaminated likelihood"
                    )
        )
        .def("SampleEvents",&PSignalContaminatedLH::SampleEvents)
        .def("EvaluateLLH",&PSignalContaminatedLH::EvaluateLLH)
        .def("SetW2Xi",&PSignalContaminatedLH::SetW2Xi)
        //.def("EnableHistogramedEvents",&PSignalContaminatedLH::EnableHistogramedEvents)
        ;

        bp::enum_<PSignalContaminatedLH::Model>("Model")
        .value("None", PSignalContaminatedLH::None)
        .value("Poisson", PSignalContaminatedLH::Poisson)
        .value("Binomial", PSignalContaminatedLH::Binomial)
        .export_values();
    }//PSignalContaminatedLH scope

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
        //.def("EnableHistogramedEvents",&ShapeLikelihood::EnableHistogramedEvents)
        .def("EnablePoissonSampling",&ShapeLikelihood::EnablePoissonSampling)
        ;

    }//ShapeLikelihood scope
    }//BinnedLikelihood scope

    }//Likelihood scope
}

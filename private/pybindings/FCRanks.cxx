#include "MLSandbox/FCRanks.h"
#include "bindingutils.h"

#include <boost/numpy.hpp>
#include <string>
#include <iostream>


namespace bn = boost::numpy;
namespace bp = boost::python;
using namespace std;
namespace mlsandbox{
    namespace python{
        static void fill(FCRanks &self, double value, bp::object obj, bool set){
            bn::ndarray arr = bn::from_object(obj,bn::dtype::get_builtin<double>(), 1, 1, bn::ndarray::CARRAY_RO);
            std::vector<double> vec(arr.get_size());
            std::copy((double*)arr.get_data(), (double*)arr.get_data()+arr.get_size(), &vec[0]);
            self.Fill(value,vec,set);
        }
        static void save(FCRanks &self, std::string file){
                mlsandbox::saveFCRanks(file,self);
        }

        static void load(FCRanks &self, std::string file){
               self=  mlsandbox::loadFCRanks(file);
        }
        static boost::python::dict get_ranks(FCRanks &self){
            std::map<double, std::vector<double> > &map = self.ranks_;
            std::map<double, std::vector<double> >::const_iterator miter;
            boost::python::dict dictionary;
            for (miter = map.begin(); miter != map.end(); ++miter) {
                boost::python::list list;
                std::vector<double>::const_iterator viter;
                for (viter = miter->second.begin(); viter != miter->second.end(); ++viter) {
                        list.append(*viter);
                }
                dictionary[miter->first] = list;
            }
            return dictionary;
        }
    }

}

void register_FCRanks()
{
  {
    bp::scope fcranks_scope =
    bp::class_<FCRanks, boost::shared_ptr<FCRanks> >
      ("FCRanks","DocString",bp::init<>(
                   "Constructor FCRanks"
                   ))
        .def("AssumeChiSqure",&FCRanks::AssumeChiSqure)
        .def("Fill",&mlsandbox::python::fill)
        .def("SetConfidenceLevel",&FCRanks::SetConfidenceLevel)
        .def("rCB",&FCRanks::rCB,
           (bp::args("xi"),
           bp::args("AssumeChiSquare")=false))
        .def("get_ranks",&mlsandbox::python::get_ranks)
        .def("save",&mlsandbox::python::save)
        .def("load",&mlsandbox::python::load)
        ;
  }
}

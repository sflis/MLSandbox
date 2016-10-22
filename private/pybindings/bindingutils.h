#include <boost/python.hpp>

#include <numpy/arrayobject.h>

#include <boost/python.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
//#include <boost/python/def_readwrite.hpp>
//#include <boost/python/def_readonly.hpp>
#include <boost/python/class.hpp>
#include <boost/python/tuple.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/back_reference.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>



#include <iostream>
namespace bp = boost::python;

// Converts a C++ map to a python dict
template <class K, class V>
boost::python::dict toPythonDict(const std::map<K, V> &map) {
    typename std::map<K, V>::const_iterator iter;
    boost::python::dict dictionary;
    for (iter = map.begin(); iter != map.end(); ++iter) {
                dictionary[iter->first] = iter->second;
    }
    return dictionary;
}


// Converts a C++ map to a python dict
template <class K, class V >
boost::python::dict toPythonDict(const std::map<K, std::vector<V> > &map) {
    typename std::map<K, std::vector<V> >::const_iterator miter;
    boost::python::dict dictionary;
    for (miter = map.begin(); miter != map.end(); ++miter) {
        boost::python::list list;
        typename std::vector<V>::const_iterator viter;
        for (viter = miter->second.begin(); viter != miter->second.end(); ++viter) {
                list.append(*viter);
        }
        dictionary[miter->first] = list;

        std::cout<<"here"<<std::endl;
    }
    return dictionary;
}


// Converts a C++ vector to a python list
template <class V>
boost::python::list toPythonList(const std::vector<V> &vector) {
    typename std::vector<V>::const_iterator iter;
    boost::python::list list;
    for (iter = vector.begin(); iter != vector.end(); ++iter) {
                list.append(*iter);
    }
    return list;
}


// Converts a python dict to a C++ map
template <class K, class V>
std::map<K, V> toStdMap(boost::python::dict dictionary) {
    typename std::map<K, V> map;
    boost::python::list list = bp::extract<boost::python::list>(dictionary.items());
    uint64_t n_items = bp::len(dictionary.items());
    for (uint64_t i = 0; i<n_items; i++) {
         bp::tuple pair = bp::extract<bp::tuple>(list[i]);
         map[bp::extract<K>(pair[0])] = bp::extract<V>(pair[0]);
    }
    return map;
}

// Converts a python list to a C++ vector
template <class V>
std::vector<V> toStdVector(boost::python::list list) {
    std::vector<V> vector;
    uint64_t n_items = bp::len(list);
    for (uint64_t i = 0; i<n_items; i++) {
        vector.push_back(bp::extract<V>(list[i]));
    }
    return vector;
}

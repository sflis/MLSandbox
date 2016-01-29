#include "MLSandbox/FCRanks.h"

#include <fstream>
#include <iostream>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/map.hpp>


//_____________________________________________________________________________
void mlsandbox::saveFCRanks(std::string file_name, FCRanks &ranks){
    std::ofstream ofs(file_name.c_str());
    boost::archive::binary_oarchive oa(ofs);
    oa<<ranks;
}
//_____________________________________________________________________________
FCRanks mlsandbox::loadFCRanks(std::string file_name){
    std::ifstream ifs(file_name.c_str());
    boost::archive::binary_iarchive ia(ifs);
    FCRanks tmp;
    ia>>tmp;
    return tmp;
}


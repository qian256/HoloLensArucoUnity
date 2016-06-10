#include "markerlabeler.h"
#include "markerlabelers/dictionary_based.h"
namespace aruco{
cv::Ptr<MarkerLabeler> MarkerLabeler::create(Dictionary::DICT_TYPES dict_type,float error_correction_rate)throw (cv::Exception)
{
    auto dict=Dictionary::loadPredefined(dict_type);
    DictionaryBased *db=new DictionaryBased();
    db->setParams(dict,error_correction_rate);
    return db;

}


cv::Ptr<MarkerLabeler> MarkerLabeler::create(std::string detector,std::string params)throw (cv::Exception){


    
    auto dict=Dictionary::loadPredefined(detector);
    DictionaryBased *db=new DictionaryBased();
    db->setParams(dict,0);
    return db;

    throw cv::Exception( -1,"No valid labeler indicated:"+detector,"Detector::create"," ",-1 );

}


}

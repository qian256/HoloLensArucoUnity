/*****************************
Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
********************************/
#ifndef _Aruco_MarkerMap_h
#define _Aruco_MarkerMap_h
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>
#include "exports.h"
#include "marker.h"
using namespace std;
namespace aruco {
/**
 * 3d representation of a marker
 */
struct ARUCO_EXPORTS Marker3DInfo : public vector< cv::Point3f > {
    Marker3DInfo() {}
    Marker3DInfo(int _id) { id = _id; }
    bool operator==(const Marker3DInfo &MI) {return id==MI.id;}
    int id; // maker id

    //returns the distance of the marker side
    float getMarkerSize()const{return cv::norm(at(0)-at(1));}
public:
    void toStream(std::ostream &str){str<<id<<" "<<size()<<" ";for(int i=0;i<size();i++) str<<at(i).x<<" "<<at(i).y<<" "<<at(i).z<<" ";}
    void fromStream(std::istream &str){int s;str>>id>>s;resize(s);for(int i=0;i<size();i++) str>>at(i).x>>at(i).y>>at(i).z;}

};

/**\brief This class defines a set of markers whose locations are attached to a common reference system, i.e., they do not move wrt each other.
 * A MarkerMap contains several markers so that they are more robustly detected.
 *
 * A MarkerMap is only a list  of the id of the markers along with the position of their corners.
 * A MarkerMap may have information about the dictionary the markers belongs to @see getDictionary()
 *
 * The position of the corners can be specified either in pixels (in a non-specific size) or in meters.
 * The first is the typical case in which you generate the image of  board  and the print it. Since you do not know in advance the real
 * size of the markers, their corners are specified in pixels, and then, the translation to meters can be made once you know the real size.
 *
 * On the other hand, you may want to have the information of your boards in meters. The MarkerMap allows you to do so.
 *
 * The point is in the mInfoType variable. It can be either PIX or METERS according to your needs.
 *
*/


class ARUCO_EXPORTS MarkerMap : public vector< Marker3DInfo > {


public:

    /**
     */
    MarkerMap();

    /**Loads from file
     * @param filePath to the config file
     */
    MarkerMap(string filePath) throw(cv::Exception);

    /**Indicates if the corners are expressed in meters
     */
    bool isExpressedInMeters() const { return mInfoType == METERS; }
    /**Indicates if the corners are expressed in meters
     */
    bool isExpressedInPixels() const { return mInfoType == PIX; }
    /**converts the passed board into meters
     */
    MarkerMap convertToMeters( float markerSize)throw (cv::Exception);
    //simple way of knowing which elements detected in an image are from this markermap
    //returns the indices of the elements in the vector 'markers' that belong to this set
    //Example: The set has the elements with ids 10,21,31,41,92
    //The input vector has the markers with ids 10,88,9,12,41
    //function returns {0,4}, because element 0 (10) of the vector belongs to the set, and also element 4 (41) belongs to the set
    std::vector<int> getIndices(vector<aruco::Marker> &markers);

    /**Returns the Info of the marker with id specified. If not in the set, throws exception
     */
    const Marker3DInfo &getMarker3DInfo(int id) const throw(cv::Exception);

    /**Returns the index of the marker (in this object) with id indicated, if is in the vector
     */
    int getIndexOfMarkerId(int id) const;
    /**Set in the list passed the set of the ids
     */
    void getIdList(vector< int > &ids, bool append = true) const;


    /**Returns an image of this to be printed. This object must be in pixels @see isExpressedInPixels(). If not,please provide the METER2PIX conversion parameter
        */
    cv::Mat getImage(float METER2PIX=0)const throw (cv::Exception);


    /**Saves the board info to a file
    */
    void saveToFile(string sfile) throw(cv::Exception);
    /**Reads board info from a file
    */
    void readFromFile(string sfile) throw(cv::Exception);



    //returns string indicating the dictionary
    std::string getDictionary()const{return dictionary;}


    enum Marker3DInfoType {
        NONE = -1,
        PIX = 0,
        METERS = 1
    }; // indicates if the data in MakersInfo is expressed in meters or in pixels so as to do conversion internally
    //returns string indicating the dictionary
    void setDictionary(std::string  d){dictionary=d;}


    // variable indicates if the data in MakersInfo is expressed in meters or in pixels so as to do conversion internally
    int mInfoType;
private:
    //dictionary it belongs to (if any)
    std::string dictionary;


private:
    /**Saves the board info to a file
    */
    void saveToFile(cv::FileStorage &fs) throw(cv::Exception);
    /**Reads board info from a file
    */
    void readFromFile(cv::FileStorage &fs) throw(cv::Exception);
public:
    void toStream(std::ostream &str);
    void fromStream(std::istream &str);
};

}

#endif

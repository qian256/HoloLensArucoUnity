/*****************************
Copyright 2016 Rafael Muñoz Salinas. All rights reserved.

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
#ifndef ARUCO_POSETRACKER
#define ARUCO_POSETRACKER
#include "exports.h"
#include <opencv2/core/core.hpp>
#include "marker.h"
#include "markermap.h"
#include "cameraparameters.h"
#include <map>
namespace aruco{

/**Tracks the position of a marker. Instead of trying to calculate the position from scratch everytime, it uses past observations to
 * estimate the pose. It should solve the problem with ambiguities that arises in some circumstances
 */
class ARUCO_EXPORTS MarkerPoseTracker{
  public:
    void estimatePose(  Marker &m,const  CameraParameters &cam_params,float markerSize);

    //returns the 4x4 transform matrix. Returns an empty matrix if last call to estimatePose returned false
    cv::Mat getRTMatrix()const;
    //return the rotation vector. Returns an empty matrix if last call to estimatePose returned false
    const cv::Mat getRvec()const{return _rvec;}
    //return the translation vector. Returns an empty matrix if last call to estimatePose returned false
    const cv::Mat getTvec()const{return _tvec;}

  private:
    cv::Mat _rvec,_tvec;//current poses
     double  solve_pnp(const std::vector<cv::Point3f> & p3d,const std::vector<cv::Point2f> & p2d,const cv::Mat &cam_matrix,const cv::Mat &dist,cv::Mat &r_io,cv::Mat &t_io);
    vector<cv::Point3f>   getMarkerPoints(float size );

};
/**Tracks the position of a markermap
 */

class ARUCO_EXPORTS MarkerMapPoseTracker{

public:
    MarkerMapPoseTracker();
    //Sets the parameters required for operation
    //If the msconf has data expressed in meters, then the markerSize parameter is not required. If it is in pixels, the markersize will be used to
    //transform to meters
    //Throws exception if wrong configuraiton
    void setParams(const  CameraParameters &cam_params,const MarkerMap &msconf, float markerSize=-1)throw(cv::Exception);
    //indicates if the call to setParams has been successfull and this object is ready to call estimatePose
    bool isValid()const{return _isValid;}
    //estimates camera pose wrt the markermap
    //returns true if pose has been obtained and false otherwise
    bool estimatePose(const  vector<Marker> &v_m);

    //returns the 4x4 transform matrix. Returns an empty matrix if last call to estimatePose returned false
    cv::Mat getRTMatrix()const;
    //return the rotation vector. Returns an empty matrix if last call to estimatePose returned false
    const cv::Mat getRvec()const{return _rvec;}
    //return the translation vector. Returns an empty matrix if last call to estimatePose returned false
    const cv::Mat getTvec()const{return _tvec;}
private:

    cv::Mat _rvec,_tvec;//current poses
    aruco::CameraParameters _cam_params;
    MarkerMap _msconf;
    std::map<int,Marker3DInfo> _map_mm;
    bool _isValid;
};

};

#endif


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
#include "markerdetector.h"
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <valarray>
#include "ar_omp.h"
#include "markerlabeler.h"
using namespace std;
using namespace cv;

namespace aruco {

/************************************
 *
 *
 *
 *
 ************************************/
MarkerDetector::MarkerDetector() {

    markerIdDetector = aruco::MarkerLabeler::create(Dictionary::ARUCO);
  //  markerIdDetector = aruco::MarkerLabeler::create("ARUCO");
    if (markerIdDetector->getBestInputSize()!=-1)setWarpSize(markerIdDetector->getBestInputSize());

}


/************************************
 *
 *
 *
 *
 ************************************/

MarkerDetector::~MarkerDetector() {}



/************************************
 *
 *
 *
 *
 ************************************/

std::vector<aruco::Marker> MarkerDetector::detect(const cv::Mat &input ) throw(cv::Exception) {
    std::vector< Marker >  detectedMarkers;
    detect(input,detectedMarkers);
    return detectedMarkers;
}

std::vector<aruco::Marker> MarkerDetector::detect(const cv::Mat &input,const CameraParameters &camParams, float markerSizeMeters , bool setYPerperdicular ) throw(cv::Exception){
    std::vector< Marker >  detectedMarkers;
    detect(input,detectedMarkers,camParams,markerSizeMeters,setYPerperdicular);
    return detectedMarkers;

}

/************************************
 *
 *
 *
 *
 ************************************/
void MarkerDetector::detect(const cv::Mat &input, std::vector< Marker > &detectedMarkers, CameraParameters camParams, float markerSizeMeters,
                            bool setYPerpendicular) throw(cv::Exception) {
    if ( camParams.CamSize!=input.size() && camParams.isValid() && markerSizeMeters>0){
        //must resize camera parameters if we want to compute properly marker poses
        CameraParameters cp_aux=camParams;
        cp_aux.resize(input.size());
        detect(input, detectedMarkers, cp_aux.CameraMatrix, cp_aux.Distorsion, markerSizeMeters, setYPerpendicular);
    }
    else{
        detect(input, detectedMarkers, camParams.CameraMatrix, camParams.Distorsion, markerSizeMeters, setYPerpendicular);
    }

}


/************************************
 *
 * Main detection function. Performs all steps
 *
 *
 ************************************/
void MarkerDetector::detect(const cv::Mat &input, vector< Marker > &detectedMarkers, Mat camMatrix, Mat distCoeff, float markerSizeMeters,
                            bool setYPerpendicular) throw(cv::Exception) {

    // it must be a 3 channel image
    if (input.type() == CV_8UC3)
        cv::cvtColor(input, grey, CV_BGR2GRAY);
    else
        grey = input;

    imagePyramid.clear();
    imagePyramid.push_back(grey);
    while(imagePyramid.back().cols>120){
      cv::Mat pyrd;
      cv::pyrDown(imagePyramid.back(),pyrd);
      imagePyramid.push_back(pyrd);
    };

    double t1 = cv::getTickCount();
    //     cv::cvtColor(grey,_ssImC ,CV_GRAY2BGR); //DELETE

    // clear input data
    detectedMarkers.clear();


    cv::Mat imgToBeThresHolded = grey;

    /// Do threshold the image and detect contours
    // work simultaneouly in a range of values of the first threshold
    int n_param1 = 2 * _params._thresParam1_range + 1;
    vector< cv::Mat > thres_images;


    //compute the different values of param1

    vector<int> p1_values;
    for(int i=std::max(3.,_params._thresParam1-2*_params._thresParam1_range);i<=_params._thresParam1+2*_params._thresParam1_range;i+=2)p1_values.push_back(i);
    thres_images.resize(p1_values.size());
#pragma omp parallel for
    for (int i = 0; i < p1_values.size(); i++)
        thresHold(_params._thresMethod, imgToBeThresHolded, thres_images[i], p1_values[i], _params._thresParam2);
    thres = thres_images[n_param1 / 2];
    //


    double t2 = cv::getTickCount();
    // find all rectangles in the thresholdes image
    vector< MarkerCandidate > MarkerCanditates;
    detectRectangles(thres_images, MarkerCanditates);

    double t3 = cv::getTickCount();

    float desiredarea=_params._markerWarpSize*_params._markerWarpSize;
    /// identify the markers
    vector< vector< Marker > > markers_omp(omp_get_max_threads());
    vector< vector< std::vector< cv::Point2f > > > candidates_omp(omp_get_max_threads());
//    for(int i=0;i<imagePyramid.size();i++){
//        string name="im"+std::to_string(i)+".jpg";
//        cv::imwrite(name,imagePyramid[i]);
//    }
#pragma omp parallel for
    for (int i = 0; i < MarkerCanditates.size(); i++) {
         // Find proyective homography
        Mat canonicalMarker;
        bool resW = false;
        //warping is one of the most time consuming operations, especially when the region is large.
        //To reduce computing time, let us find in the image pyramid, the best configuration to save time
        //indicates how much bigger observation is wrt to desired patch
        int imgPyrIdx=0;
        for(int p=1;p<imagePyramid.size();p++){
            if (MarkerCanditates[i].getArea() / pow(4,p) >= desiredarea ) imgPyrIdx=p;
            else break;
        }

        vector<cv::Point2f> points2d_pyr=MarkerCanditates[i];
        for(auto &p:points2d_pyr) p*=1./pow(2,imgPyrIdx);
        resW = warp(imagePyramid[imgPyrIdx], canonicalMarker, Size(_params._markerWarpSize, _params._markerWarpSize), points2d_pyr);
        //go to a pyramid that minimizes the ratio

        if (resW) {
            int id,nRotations;
            if (markerIdDetector->detect(canonicalMarker, id,nRotations)) {
                 if (_params._cornerMethod == LINES) // make LINES refinement before lose contour points
                    refineCandidateLines(MarkerCanditates[i], camMatrix, distCoeff);
                markers_omp[omp_get_thread_num()].push_back(MarkerCanditates[i]);
                markers_omp[omp_get_thread_num()].back().id = id;
                // sort the points so that they are always in the same order no matter the camera orientation
                std::rotate(markers_omp[omp_get_thread_num()].back().begin(), markers_omp[omp_get_thread_num()].back().begin() + 4 - nRotations, markers_omp[omp_get_thread_num()].back().end());
              } else
                candidates_omp[omp_get_thread_num()].push_back(MarkerCanditates[i]);
        }
    }
    // unify parallel data
    joinVectors(markers_omp, detectedMarkers, true);
    joinVectors(candidates_omp, _candidates, true);





    double t4 = cv::getTickCount();

    /// refine the corner location if desired
    if (detectedMarkers.size() > 0 && _params._cornerMethod != NONE && _params._cornerMethod != LINES) {

        vector< Point2f > Corners;
        for (unsigned int i = 0; i < detectedMarkers.size(); i++)
            for (int c = 0; c < 4; c++)
                Corners.push_back(detectedMarkers[i][c]);


          if (_params._cornerMethod == SUBPIX) {
            cornerSubPix(grey, Corners, cvSize(_params._subpix_wsize, _params._subpix_wsize), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 12, 0.005));
        }
        // copy back
        for (unsigned int i = 0; i < detectedMarkers.size(); i++)
            for (int c = 0; c < 4; c++)
                detectedMarkers[i][c] = Corners[i * 4 + c];
    }

    double t5 = cv::getTickCount();

    // sort by id
    std::sort(detectedMarkers.begin(), detectedMarkers.end());
     // there might be still the case that a marker is detected twice because of the double border indicated earlier,
    // detect and remove these cases
    vector< bool > toRemove(detectedMarkers.size(), false);

    for (int i = 0; i < int(detectedMarkers.size()) - 1; i++) {
        for (int j = i+1; j < int(detectedMarkers.size()) && !toRemove[i] ; j++) {
            if (detectedMarkers[i].id == detectedMarkers[j].id) {
                // deletes the one with smaller perimeter
                if (perimeter(detectedMarkers[i]) < perimeter(detectedMarkers[j]))
                    toRemove[i  ] = true;
                else
                    toRemove[j  ] = true;

            }
        }
    }


    // remove markers with corners too near the image limits
    int borderDistThresX = _params._borderDistThres * float(input.cols);
    int borderDistThresY = _params._borderDistThres * float(input.rows);
    for (size_t i = 0; i < detectedMarkers.size(); i++) {
        // delete if any of the corners is too near image border
        for (size_t c = 0; c < detectedMarkers[i].size(); c++) {
            if (detectedMarkers[i][c].x < borderDistThresX || detectedMarkers[i][c].y < borderDistThresY ||
                detectedMarkers[i][c].x > input.cols - borderDistThresX || detectedMarkers[i][c].y > input.rows - borderDistThresY) {
                toRemove[i] = true;
            }
        }
    }


    // remove the markers marker
    removeElements(detectedMarkers, toRemove);

    /// detect the position of detected markers if desired
    if (camMatrix.rows != 0 && markerSizeMeters > 0) {
        for (unsigned int i = 0; i < detectedMarkers.size(); i++)
            detectedMarkers[i].calculateExtrinsics(markerSizeMeters, camMatrix, distCoeff, setYPerpendicular);
    }
    double t6 = cv::getTickCount();

//    cerr << "Threshold: " << 1000*(t2 - t1) / double(cv::getTickFrequency()) << endl;
//    cerr << "Rectangles: " << 1000*(t3 - t2) / double(cv::getTickFrequency()) << endl;
//    cerr << "Identify: " << 1000*(t4 - t3) / double(cv::getTickFrequency()) << endl;
//    cerr << "Subpixel: " << 1000*(t5 - t4) / double(cv::getTickFrequency()) << endl;
//    cerr << "Filtering: " << 1000*(t6 - t5) / double(cv::getTickFrequency()) << endl;
}
struct   PointSet_2D:public std::vector<cv::Point2f>
{
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return   size(); }
    // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
    inline float kdtree_distance(const float *p1, const size_t idx_p2,size_t size) const
    {
        const float d0=p1[0]-at(idx_p2).x;
        const float d1=p1[1]-at(idx_p2).y;
        return d0*d0+d1*d1;
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline float kdtree_get_pt(const size_t idx, int dim) const
    {
         return  dim==0? at(idx).x:at(idx).y;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX &bb) const { return false; }
};


/************************************
 *
 * Crucial step. Detects the rectangular regions of the thresholded image
 *
 *
 ************************************/
void MarkerDetector::detectRectangles(const cv::Mat &thres, vector< std::vector< cv::Point2f > > &MarkerCanditates) {
    vector< MarkerCandidate > candidates;
    vector< cv::Mat > thres_v;
    thres_v.push_back(thres);
    detectRectangles(thres_v, candidates);
    // create the output
    MarkerCanditates.resize(candidates.size());
    for (size_t i = 0; i < MarkerCanditates.size(); i++)
        MarkerCanditates[i] = candidates[i];
}

void MarkerDetector::detectRectangles(vector< cv::Mat > &thresImgv, vector< MarkerCandidate > &OutMarkerCanditates) {
    //         omp_set_num_threads ( 1 );
    vector< vector< MarkerCandidate > > MarkerCanditatesV(omp_get_max_threads());
    // calcualte the min_max contour sizes
    //int minSize = _params._minSize * std::max(thresImgv[0].cols, thresImgv[0].rows) * 4;
    int maxSize = _params._maxSize * std::max(thresImgv[0].cols, thresImgv[0].rows) * 4;
    int minSize=  std::min ( float(_params._minSize_pix) , _params._minSize* std::max(thresImgv[0].cols, thresImgv[0].rows) * 4 );
//         cv::Mat input;
//         cv::cvtColor ( thresImgv[0],input,CV_GRAY2BGR );
#pragma omp parallel for
    for (int img_idx = 0; img_idx < thresImgv.size(); img_idx++) {
        std::vector< cv::Vec4i > hierarchy2;
        std::vector< std::vector< cv::Point > > contours2;
        cv::Mat thres2;
        thresImgv[img_idx].copyTo(thres2);
        cv::findContours(thres2, contours2, hierarchy2, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

        vector< Point > approxCurve;
        /// for each contour, analyze if it is a paralelepiped likely to be the marker
        for (unsigned int i = 0; i < contours2.size(); i++) {

            // check it is a possible element by first checking is has enough points
            if (minSize < contours2[i].size() && contours2[i].size() < maxSize) {
                // approximate to a poligon
                approxPolyDP(contours2[i], approxCurve, double(contours2[i].size()) * 0.05, true);
                // 				drawApproxCurve(copy,approxCurve,Scalar(0,0,255));
                // check that the poligon has 4 points
                if (approxCurve.size() == 4) {
                    /*
                                            drawContour ( input,contours2[i],Scalar ( 255,0,225 ) );
                                            namedWindow ( "input" );
                                            imshow ( "input",input );*/
                    //  	 	waitKey(0);
                    // and is convex
                    if (isContourConvex(Mat(approxCurve))) {
                        // 					      drawApproxCurve(input,approxCurve,Scalar(255,0,255));
                        // 						//ensure that the   distace between consecutive points is large enough
                        float minDist = 1e10;
                        for (int j = 0; j < 4; j++) {
                            float d = std::sqrt((float)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) * (approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                                                (approxCurve[j].y - approxCurve[(j + 1) % 4].y) * (approxCurve[j].y - approxCurve[(j + 1) % 4].y));
                            // 		norm(Mat(approxCurve[i]),Mat(approxCurve[(i+1)%4]));
                            if (d < minDist) minDist = d;
                        }
                        // check that distance is not very small
                        if (minDist > 10) {
                            // add the points
                            // 	      cout<<"ADDED"<<endl;
                            MarkerCanditatesV[omp_get_thread_num()].push_back(MarkerCandidate());
                            MarkerCanditatesV[omp_get_thread_num()].back().idx = i;
                            if (_params._cornerMethod==LINES)//save all contour points if you need lines refinement method
                                MarkerCanditatesV[omp_get_thread_num()].back().contour = contours2[i];
                            for (int j = 0; j < 4; j++)
                                MarkerCanditatesV[omp_get_thread_num()].back().push_back(Point2f(approxCurve[j].x, approxCurve[j].y));
                        }
                    }
                }
            }
        }
    }

    // join all candidates
    vector< MarkerCandidate > MarkerCanditates;

    for (size_t i = 0; i < MarkerCanditatesV.size(); i++)
        for (size_t j = 0; j < MarkerCanditatesV[i].size(); j++) {
            MarkerCanditates.push_back(MarkerCanditatesV[i][j]);
        }

    /// sort the points in anti-clockwise order
    valarray< bool > swapped(false, MarkerCanditates.size()); // used later
    for (unsigned int i = 0; i < MarkerCanditates.size(); i++) {

        // trace a line between the first and second point.
        // if the thrid point is at the right side, then the points are anti-clockwise
        double dx1 = MarkerCanditates[i][1].x - MarkerCanditates[i][0].x;
        double dy1 = MarkerCanditates[i][1].y - MarkerCanditates[i][0].y;
        double dx2 = MarkerCanditates[i][2].x - MarkerCanditates[i][0].x;
        double dy2 = MarkerCanditates[i][2].y - MarkerCanditates[i][0].y;
        double o = (dx1 * dy2) - (dy1 * dx2);

        if (o < 0.0) { // if the third point is in the left side, then sort in anti-clockwise order
            swap(MarkerCanditates[i][1], MarkerCanditates[i][3]);
            swapped[i] = true;
            // sort the contour points
            //  	    reverse(MarkerCanditates[i].contour.begin(),MarkerCanditates[i].contour.end());//????
        }
    }
    /// remove these elements which corners are too close to each other
    // first detect candidates to be removed
    vector< vector< pair< int, int > > > TooNearCandidates_omp(omp_get_max_threads());
#pragma omp parallel for
    for (unsigned int i = 0; i < MarkerCanditates.size(); i++) {
        // calculate the average distance of each corner to the nearest corner of the other marker candidate
        for (unsigned int j = i + 1; j < MarkerCanditates.size(); j++) {
            valarray< float > vdist(4);
            for (int c = 0; c < 4; c++)
                vdist[c] = sqrt((MarkerCanditates[i][c].x - MarkerCanditates[j][c].x) * (MarkerCanditates[i][c].x - MarkerCanditates[j][c].x) +
                                (MarkerCanditates[i][c].y - MarkerCanditates[j][c].y) * (MarkerCanditates[i][c].y - MarkerCanditates[j][c].y));
            //                 dist/=4;
            // if distance is too small
            if (vdist[0] < 6 && vdist[1] < 6 && vdist[2] < 6 && vdist[3] < 6) {
                TooNearCandidates_omp[omp_get_thread_num()].push_back(pair< int, int >(i, j));
            }
        }
    }


    // join
    vector< pair< int, int > > TooNearCandidates;
    joinVectors(TooNearCandidates_omp, TooNearCandidates);
    // mark for removal the element of  the pair with smaller perimeter
    valarray< bool > toRemove(false, MarkerCanditates.size());
    for (unsigned int i = 0; i < TooNearCandidates.size(); i++) {
        if (perimeter(MarkerCanditates[TooNearCandidates[i].first]) > perimeter(MarkerCanditates[TooNearCandidates[i].second]))
            toRemove[TooNearCandidates[i].second] = true;
        else
            toRemove[TooNearCandidates[i].first] = true;
    }

    // remove the invalid ones
    // finally, assign to the remaining candidates the contour
    OutMarkerCanditates.reserve(MarkerCanditates.size());
    for (size_t i = 0; i < MarkerCanditates.size(); i++) {
        if (!toRemove[i]) {
            OutMarkerCanditates.push_back(MarkerCanditates[i]);
            //                 OutMarkerCanditates.back().contour=contours2[ MarkerCanditates[i].idx];
            if (swapped[i] && OutMarkerCanditates.back().contour.size()>1) // if the corners where swapped, it is required to reverse here the points so that they are in the same order
                reverse(OutMarkerCanditates.back().contour.begin(), OutMarkerCanditates.back().contour.end()); //????
        }
    }

    /*
            for ( size_t i=0; i<OutMarkerCanditates.size(); i++ )
                    OutMarkerCanditates[i].draw ( input,cv::Scalar ( 124,  255,125 ) );


            namedWindow ( "input" );
            imshow ( "input",input );*/
}

/************************************
 *
 *
 * attempt to beat adaptiveThreshold by precomputing the integral image for all the possibilities
 * I did not make it yet
 *
 ************************************/

void  MarkerDetector::adpt_threshold_multi( const Mat &grey, std::vector<Mat> &outThresImages,double param1  ,double param1_range , double param2,double param2_range ){

//    param2_range=2;
    int start_p1 = std::max(3.,param1-2*param1_range);
    int end_p1 = param1+2*param1_range;
    int start_p2 = std::max(3.,param2-2*param2_range);
    int end_p2 = param2+2*param2_range;
    vector<std::pair<int,int> > p1_2_values;
    for(int i=start_p1;i<=end_p1;i+=2)
        for(int j=start_p2;j<=end_p2;j+=2)
            p1_2_values.push_back(std::pair<int,int>(i,j));
    outThresImages.resize(p1_2_values.size());

    cv::Mat intimg;
    cv::integral(grey,intimg);
    //now, run in parallel creating the thresholded images
#pragma omp parallel for
    for(int i=0;i<p1_2_values.size();i++){
      //  cout<<p1_2_values[i].first<<" "<<p1_2_values[i].second<<endl;
        //now, for each image, apply the
        float inv_area=1./(p1_2_values[i].first*p1_2_values[i].first);
        int wsize_2=p1_2_values[i].first/2;
        outThresImages[i].create(grey.size(),grey.type() );
        //start moving accross the image
        for(int y=wsize_2;y<grey.rows-wsize_2;y++){
            int *_y1=intimg.ptr<int>(y-wsize_2);
            int *_y2=intimg.ptr<int>(y+wsize_2+1);
            uchar *out=      outThresImages[i].ptr<uchar>(y);
            for(int x=wsize_2;x<grey.cols-wsize_2;x++){
                int x2=x+wsize_2+1;
                int x1=x-wsize_2;
               // int sum=intimg.at<int>(y2,x2)-intimg.at<int>(y2,x1)-intimg.at<int>(y1,x2)+intimg.at<int>(y1,x1);
                float mean=float( _y2[x2]-_y2[x1]-_y1[x2]+_y1[x1])* inv_area ;
                if ( mean- grey.at<uchar>(y,x)>p1_2_values[i].second)
                    out[x]=255;
                else
                    out[x]=0;
            }
        }
    }

}

/************************************
 *
 *
 *
 *
 ************************************/
void MarkerDetector::thresHold(int method, const Mat &grey, Mat &out, double param1, double param2) throw(cv::Exception) {

    if (param1 == -1)
        param1 = _params._thresParam1;
    if (param2 == -1)
        param2 = _params._thresParam2;

    if (grey.type() != CV_8UC1)
        throw cv::Exception(9001, "grey.type()!=CV_8UC1", "MarkerDetector::thresHold", __FILE__, __LINE__);
    switch (method) {
    case FIXED_THRES:
        cv::threshold(grey, out, param1, 255, CV_THRESH_BINARY_INV);
        break;
    case ADPT_THRES: // currently, this is the best method
        // ensure that _thresParam1%2==1
        if (param1 < 3)
            param1 = 3;
        else if (((int)param1) % 2 != 1)
            param1 = (int)(param1 + 1);

        cv::adaptiveThreshold(grey, out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, param1, param2);
        break;
    case CANNY: {
        // this should be the best method, and generally it is.
        // However, some times there are small holes in the marker contour that makes
        // the contour detector not to find it properly
        // if there is a missing pixel
        cv::Canny(grey, out, 10, 220);
        // I've tried a closing but it add many more points that some
        // times makes this even worse
        // 			  Mat aux;
        // 			  cv::morphologyEx(thres,aux,MORPH_CLOSE,Mat());
        // 			  out=aux;
    } break;
    }
}
/************************************
 *
 *
 *
 *
 ************************************/
bool MarkerDetector::warp(Mat &in, Mat &out, Size size, vector< Point2f > points) throw(cv::Exception) {

    if (points.size() != 4)
        throw cv::Exception(9001, "point.size()!=4", "MarkerDetector::warp", __FILE__, __LINE__);
    // obtain the perspective transform
    Point2f pointsRes[4], pointsIn[4];
    for (int i = 0; i < 4; i++)
        pointsIn[i] = points[i];
    pointsRes[0] = (Point2f(0, 0));
    pointsRes[1] = Point2f(size.width - 1, 0);
    pointsRes[2] = Point2f(size.width - 1, size.height - 1);
    pointsRes[3] = Point2f(0, size.height - 1);
    Mat M = getPerspectiveTransform(pointsIn, pointsRes);
    cv::warpPerspective(in, out, M, size, cv::INTER_NEAREST);
    return true;
}

void findCornerPointsInContour(const vector< cv::Point2f > &points, const vector< cv::Point > &contour, vector< int > &idxs) {
    assert(points.size() == 4);
    int idxSegments[4] = {-1, -1, -1, -1};
    // the first point coincides with one
    cv::Point points2i[4];
    for (int i = 0; i < 4; i++) {
        points2i[i].x = points[i].x;
        points2i[i].y = points[i].y;
    }

    for (size_t i = 0; i < contour.size(); i++) {
        if (idxSegments[0] == -1)
            if (contour[i] == points2i[0])
                idxSegments[0] = i;
        if (idxSegments[1] == -1)
            if (contour[i] == points2i[1])
                idxSegments[1] = i;
        if (idxSegments[2] == -1)
            if (contour[i] == points2i[2])
                idxSegments[2] = i;
        if (idxSegments[3] == -1)
            if (contour[i] == points2i[3])
                idxSegments[3] = i;
    }
    idxs.resize(4);
    for (int i = 0; i < 4; i++)
        idxs[i] = idxSegments[i];
}

int findDeformedSidesIdx(const vector< cv::Point > &contour, const vector< int > &idxSegments) {
    float distSum[4] = {0, 0, 0, 0};
    cv::Scalar colors[4] = {cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(111, 111, 0)};

    for (int i = 0; i < 3; i++) {
        cv::Point p1 = contour[idxSegments[i]];
        cv::Point p2 = contour[idxSegments[i + 1]];
        float inv_den = 1. / sqrt(float((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y)));
        //   d=|v^^·r|=(|(x_2-x_1)(y_1-y_0)-(x_1-x_0)(y_2-y_1)|)/(sqrt((x_2-x_1)^2+(y_2-y_1)^2)).
        //         cerr<<"POSS="<<idxSegments[i]<<" "<<idxSegments[i+1]<<endl;
        for (size_t j = idxSegments[i]; j < idxSegments[i + 1]; j++) {
            float dist = std::fabs(float((p2.x - p1.x) * (p1.y - contour[j].y) - (p1.x - contour[j].x) * (p2.y - p1.y))) * inv_den;
            distSum[i] += dist;
            //             cerr<< dist<<" ";
            //             cv::rectangle(_ssImC,contour[j],contour[j],colors[i],-1);
        }
        distSum[i] /= float(idxSegments[i + 1] - idxSegments[i]);
        //         cout<<endl<<endl;
    }


    // for the last one
    cv::Point p1 = contour[idxSegments[0]];
    cv::Point p2 = contour[idxSegments[3]];
    float inv_den = 1. / std::sqrt(float((p2.x - p1.x) * (p2.x - p1.x) + (p2.y - p1.y) * (p2.y - p1.y)));
    //   d=|v^^·r|=(|(x_2-x_1)(y_1-y_0)-(x_1-x_0)(y_2-y_1)|)/(sqrt((x_2-x_1)^2+(y_2-y_1)^2)).
    for (size_t j = 0; j < idxSegments[0]; j++)
        distSum[3] += std::fabs(float((p2.x - p1.x) * (p1.y - contour[j].y) - (p1.x - contour[j].x) * (p2.y - p1.y))) * inv_den;
    for (size_t j = idxSegments[3]; j < contour.size(); j++)
        distSum[3] += std::fabs(float((p2.x - p1.x) * (p1.y - contour[j].y) - (p1.x - contour[j].x) * (p2.y - p1.y))) * inv_den;

    distSum[3] /= float(idxSegments[0] + (contour.size() - idxSegments[3]));
    // now, get the maximum
    /*    for (int i=0;i<4;i++)
            cout<<"DD="<<distSum[i]<<endl;*/
    // check the two combinations to see the one with higher error
    if (distSum[0] + distSum[2] > distSum[1] + distSum[3])
        return 0;
    else
        return 1;
}

void setPointIntoImage(cv::Point2f &p, cv::Size s) {
    if (p.x < 0)
        p.x = 0;
    else if (p.x >= s.width)
        p.x = s.width - 1;
    if (p.y < 0)
        p.y = 0;
    else if (p.y >= s.height)
        p.y = s.height - 1;
}

void setPointIntoImage(cv::Point &p, cv::Size s) {
    if (p.x < 0)
        p.x = 0;
    else if (p.x >= s.width)
        p.x = s.width - 1;
    if (p.y < 0)
        p.y = 0;
    else if (p.y >= s.height)
        p.y = s.height - 1;
}
/************************************
 *
 *
 *
 *
 ************************************/
bool MarkerDetector::warp_cylinder(Mat &in, Mat &out, Size size, MarkerCandidate &mcand) throw(cv::Exception) {

    if (mcand.size() != 4)
        throw cv::Exception(9001, "point.size()!=4", "MarkerDetector::warp", __FILE__, __LINE__);

    // check first the real need for cylinder warping
    //     cout<<"im="<<mcand.contour.size()<<endl;

    //     for (size_t i=0;i<mcand.contour.size();i++) {
    //         cv::rectangle(_ssImC ,mcand.contour[i],mcand.contour[i],cv::Scalar(111,111,111),-1 );
    //     }
    //     mcand.draw(imC,cv::Scalar(0,255,0));
    // find the 4 different segments of the contour
    vector< int > idxSegments;
    findCornerPointsInContour(mcand, mcand.contour, idxSegments);
    // let us rearrange the points so that the first corner is the one whith smaller idx
    int minIdx = 0;
    for (int i = 1; i < 4; i++)
        if (idxSegments[i] < idxSegments[minIdx])
            minIdx = i;
    // now, rotate the points to be in this order
    std::rotate(idxSegments.begin(), idxSegments.begin() + minIdx, idxSegments.end());
    std::rotate(mcand.begin(), mcand.begin() + minIdx, mcand.end());

    //     cout<<"idxSegments="<<idxSegments[0]<< " "<<idxSegments[1]<< " "<<idxSegments[2]<<" "<<idxSegments[3]<<endl;
    // now, determine the sides that are deformated by cylinder perspective
    int defrmdSide = findDeformedSidesIdx(mcand.contour, idxSegments);
    //     cout<<"Def="<<defrmdSide<<endl;

    // instead of removing perspective distortion  of the rectangular region
    // given by the rectangle, we enlarge it a bit to include the deformed parts
    cv::Point2f center = mcand.getCenter();
    Point2f enlargedRegion[4];
    for (int i = 0; i < 4; i++)
        enlargedRegion[i] = mcand[i];
    if (defrmdSide == 0) {
        enlargedRegion[0] = mcand[0] + (mcand[3] - mcand[0]) * 1.2;
        enlargedRegion[1] = mcand[1] + (mcand[2] - mcand[1]) * 1.2;
        enlargedRegion[2] = mcand[2] + (mcand[1] - mcand[2]) * 1.2;
        enlargedRegion[3] = mcand[3] + (mcand[0] - mcand[3]) * 1.2;
    } else {
        enlargedRegion[0] = mcand[0] + (mcand[1] - mcand[0]) * 1.2;
        enlargedRegion[1] = mcand[1] + (mcand[0] - mcand[1]) * 1.2;
        enlargedRegion[2] = mcand[2] + (mcand[3] - mcand[2]) * 1.2;
        enlargedRegion[3] = mcand[3] + (mcand[2] - mcand[3]) * 1.2;
    }
    for (size_t i = 0; i < 4; i++)
        setPointIntoImage(enlargedRegion[i], in.size());

    /*
        cv::Scalar colors[4]={cv::Scalar(0,0,255),cv::Scalar(255,0,0),cv::Scalar(0,255,0),cv::Scalar(111,111,0)};
        for (int i=0;i<4;i++) {
            cv::rectangle(_ssImC,mcand.contour[idxSegments[i]]-cv::Point(2,2),mcand.contour[idxSegments[i]]+cv::Point(2,2),colors[i],-1 );
            cv::rectangle(_ssImC,enlargedRegion[i]-cv::Point2f(2,2),enlargedRegion[i]+cv::Point2f(2,2),colors[i],-1 );

        }*/
    //     cv::imshow("imC",_ssImC);


    // calculate the max distance from each contour point the line of the corresponding segment it belongs to
    //     calculate
    //      cv::waitKey(0);
    // check that the region is into image limits
    // obtain the perspective transform
    Point2f pointsRes[4], pointsIn[4];
    for (int i = 0; i < 4; i++)
        pointsIn[i] = mcand[i];

    cv::Size enlargedSize = size;
    enlargedSize.width += 2 * enlargedSize.width * 0.2;
    pointsRes[0] = (Point2f(0, 0));
    pointsRes[1] = Point2f(enlargedSize.width - 1, 0);
    pointsRes[2] = Point2f(enlargedSize.width - 1, enlargedSize.height - 1);
    pointsRes[3] = Point2f(0, enlargedSize.height - 1);
    // rotate to ensure that deformed sides are in the horizontal axis when warping
    if (defrmdSide == 0)
        rotate(pointsRes, pointsRes + 1, pointsRes + 4);
    cv::Mat imAux, imAux2(enlargedSize, CV_8UC1);
    Mat M = cv::getPerspectiveTransform(enlargedRegion, pointsRes);
    cv::warpPerspective(in, imAux, M, enlargedSize, cv::INTER_NEAREST);

    // now, transform all points to the new image
    vector< cv::Point > pointsCO(mcand.contour.size());
    assert(M.type() == CV_64F);
    assert(M.cols == 3 && M.rows == 3);
    //     cout<<M<<endl;
    double *mptr = M.ptr< double >(0);
    imAux2.setTo(cv::Scalar::all(0));


    for (size_t i = 0; i < mcand.contour.size(); i++) {
        float inX = mcand.contour[i].x;
        float inY = mcand.contour[i].y;
        float w = inX * mptr[6] + inY * mptr[7] + mptr[8];
        cv::Point2f pres;
        pointsCO[i].x = ((inX * mptr[0] + inY * mptr[1] + mptr[2]) / w) + 0.5;
        pointsCO[i].y = ((inX * mptr[3] + inY * mptr[4] + mptr[5]) / w) + 0.5;
        // make integers
        setPointIntoImage(pointsCO[i], imAux.size()); // ensure points are into image limits
        // 	cout<<"p="<<pointsCO[i]<<" "<<imAux.size().width<<" "<<imAux.size().height<<endl;
        imAux2.at< uchar >(pointsCO[i].y, pointsCO[i].x) = 255;
        if (pointsCO[i].y > 0)
            imAux2.at< uchar >(pointsCO[i].y - 1, pointsCO[i].x) = 255;
        if (pointsCO[i].y < imAux2.rows - 1)
            imAux2.at< uchar >(pointsCO[i].y + 1, pointsCO[i].x) = 255;
    }

    cv::Mat outIm(enlargedSize, CV_8UC1);
    outIm.setTo(cv::Scalar::all(0));
    // now, scan in lines to determine the required displacement
    for (int y = 0; y < imAux2.rows; y++) {
        uchar *_offInfo = imAux2.ptr< uchar >(y);
        int start = -1, end = -1;
        // determine the start and end of markerd regions
        for (int x = 0; x < imAux.cols; x++) {
            if (_offInfo[x]) {
                if (start == -1)
                    start = x;
                else
                    end = x;
            }
        }
        //       cout<<"S="<<start<<" "<<end<<" "<<end-start<<" "<<(size.width>>1)<<endl;
        // check that the size is big enough and
        assert(start != -1 && end != -1 && (end - start) > size.width >> 1);
        uchar *In_image = imAux.ptr< uchar >(y);
        uchar *Out_image = outIm.ptr< uchar >(y);
        memcpy(Out_image, In_image + start, imAux.cols - start);
    }


    //     cout<<"SS="<<mcand.contour.size()<<" "<<pointsCO.size()<<endl;
    // get the central region with the size specified
    cv::Mat centerReg = outIm(cv::Range::all(), cv::Range(0, size.width));
    out = centerReg.clone();
    //     cv::perspectiveTransform(mcand.contour,pointsCO,M);
    // draw them
    //     cv::imshow("out2",out);
    //     cv::imshow("imm",imAux2);
    //     cv::waitKey(0);
    return true;
}
/************************************
 *
 *
 *
 *
 ************************************/
bool MarkerDetector::isInto(Mat &contour, vector< Point2f > &b) {

    for (unsigned int i = 0; i < b.size(); i++)
        if (pointPolygonTest(contour, b[i], false) > 0)
            return true;
    return false;
}
/************************************
 *
 *
 *
 *
 ************************************/
int MarkerDetector::perimeter(vector< Point2f > &a) {
    int sum = 0;
    for (unsigned int i = 0; i < a.size(); i++) {
        int i2 = (i + 1) % a.size();
        sum += sqrt((a[i].x - a[i2].x) * (a[i].x - a[i2].x) + (a[i].y - a[i2].y) * (a[i].y - a[i2].y));
    }
    return sum;
}





/**
 *
 *
 */
void MarkerDetector::refineCandidateLines(MarkerDetector::MarkerCandidate &candidate, const cv::Mat &camMatrix, const cv::Mat &distCoeff) {
    // search corners on the contour vector
    vector< int > cornerIndex(4,-1);
    for (unsigned int j = 0; j < candidate.contour.size(); j++) {
        for (unsigned int k = 0; k < 4; k++) {
            if (candidate.contour[j].x == candidate[k].x && candidate.contour[j].y == candidate[k].y) {
                cornerIndex[k] = j;
            }
        }
    }

    // contour pixel in inverse order or not?
    bool inverse;
    if ((cornerIndex[1] > cornerIndex[0]) && (cornerIndex[2] > cornerIndex[1] || cornerIndex[2] < cornerIndex[0]))
        inverse = false;
    else if (cornerIndex[2] > cornerIndex[1] && cornerIndex[2] < cornerIndex[0])
        inverse = false;
    else
        inverse = true;


    // get pixel vector for each line of the marker
    int inc = 1;
    if (inverse)
        inc = -1;

    // undistort contour
    vector< Point2f > contour2f;
    if(!camMatrix.empty() && !distCoeff.empty()){
    for (unsigned int i = 0; i < candidate.contour.size(); i++)
        contour2f.push_back(cv::Point2f(candidate.contour[i].x, candidate.contour[i].y));
    if (!camMatrix.empty() && !distCoeff.empty())
        cv::undistortPoints(contour2f, contour2f, camMatrix, distCoeff, cv::Mat(), camMatrix);

    }
    else {
        contour2f.reserve(candidate.contour.size());
        for(auto p:candidate.contour)
            contour2f.push_back(cv::Point2f(p.x,p.y));
    }

    vector< std::vector< cv::Point2f > > contourLines;
    contourLines.resize(4);
    for (unsigned int l = 0; l < 4; l++) {
        for (int j = (int)cornerIndex[l]; j != (int)cornerIndex[(l + 1) % 4]; j += inc) {
            if (j == (int)candidate.contour.size() && !inverse)
                j = 0;
            else if (j == 0 && inverse)
                j = candidate.contour.size() - 1;
            contourLines[l].push_back(contour2f[j]);
            if (j == (int)cornerIndex[(l + 1) % 4])
                break; // this has to be added because of the previous ifs
        }
    }

    // interpolate marker lines
    vector< Point3f > lines;
    lines.resize(4);
    for (unsigned int j = 0; j < lines.size(); j++)
        interpolate2Dline(contourLines[j], lines[j]);

    // get cross points of lines
    vector< Point2f > crossPoints;
    crossPoints.resize(4);
    for (unsigned int i = 0; i < 4; i++)
        crossPoints[i] = getCrossPoint(lines[(i - 1) % 4], lines[i]);

    // distort corners again if undistortion was performed
    if (!camMatrix.empty() && !distCoeff.empty())
        distortPoints(crossPoints, crossPoints, camMatrix, distCoeff);

    // reassing points
    for (unsigned int j = 0; j < 4; j++)
        candidate[j] = crossPoints[j];
}


/**
 */
void MarkerDetector::interpolate2Dline(const std::vector< Point2f > &inPoints, Point3f &outLine) {

    float minX, maxX, minY, maxY;
    minX = maxX = inPoints[0].x;
    minY = maxY = inPoints[0].y;
    for (unsigned int i = 1; i < inPoints.size(); i++) {
        if (inPoints[i].x < minX)
            minX = inPoints[i].x;
        if (inPoints[i].x > maxX)
            maxX = inPoints[i].x;
        if (inPoints[i].y < minY)
            minY = inPoints[i].y;
        if (inPoints[i].y > maxY)
            maxY = inPoints[i].y;
    }

    // create matrices of equation system
    Mat A(inPoints.size(), 2, CV_32FC1, Scalar(0));
    Mat B(inPoints.size(), 1, CV_32FC1, Scalar(0));
    Mat X;



    if (maxX - minX > maxY - minY) {
        // Ax + C = y
        for (int i = 0; i < inPoints.size(); i++) {

            A.at< float >(i, 0) = inPoints[i].x;
            A.at< float >(i, 1) = 1.;
            B.at< float >(i, 0) = inPoints[i].y;
        }

        // solve system
        solve(A, B, X, DECOMP_SVD);
        // return Ax + By + C
        outLine = Point3f(X.at< float >(0, 0), -1., X.at< float >(1, 0));
    } else {
        // By + C = x
        for (int i = 0; i < inPoints.size(); i++) {

            A.at< float >(i, 0) = inPoints[i].y;
            A.at< float >(i, 1) = 1.;
            B.at< float >(i, 0) = inPoints[i].x;
        }

        // solve system
        solve(A, B, X, DECOMP_SVD);
        // return Ax + By + C
        outLine = Point3f(-1., X.at< float >(0, 0), X.at< float >(1, 0));
    }
}

/**
 */
Point2f MarkerDetector::getCrossPoint(const cv::Point3f &line1, const cv::Point3f &line2) {

    // create matrices of equation system
    Mat A(2, 2, CV_32FC1, Scalar(0));
    Mat B(2, 1, CV_32FC1, Scalar(0));
    Mat X;

    A.at< float >(0, 0) = line1.x;
    A.at< float >(0, 1) = line1.y;
    B.at< float >(0, 0) = -line1.z;

    A.at< float >(1, 0) = line2.x;
    A.at< float >(1, 1) = line2.y;
    B.at< float >(1, 0) = -line2.z;

    // solve system
    solve(A, B, X, DECOMP_SVD);
    return Point2f(X.at< float >(0, 0), X.at< float >(1, 0));
}


/**
 */
void MarkerDetector::distortPoints(vector< cv::Point2f > in, vector< cv::Point2f > &out, const Mat &camMatrix, const Mat &distCoeff) {
    // trivial extrinsics
    cv::Mat Rvec = cv::Mat(3, 1, CV_32FC1, cv::Scalar::all(0));
    cv::Mat Tvec = Rvec.clone();
    // calculate 3d points and then reproject, so opencv makes the distortion internally
    vector< cv::Point3f > cornersPoints3d;
    for (unsigned int i = 0; i < in.size(); i++)
        cornersPoints3d.push_back(cv::Point3f((in[i].x - camMatrix.at< float >(0, 2)) / camMatrix.at< float >(0, 0), // x
                                              (in[i].y - camMatrix.at< float >(1, 2)) / camMatrix.at< float >(1, 1), // y
                                              1)); // z
    cv::projectPoints(cornersPoints3d, Rvec, Tvec, camMatrix, distCoeff, out);
}



/************************************
 *
 *
 *
 *
 ************************************/
void MarkerDetector::drawAllContours(Mat input, std::vector< std::vector< cv::Point > > &contours) { drawContours(input, contours, -1, Scalar(255, 0, 255)); }

/************************************
 *
 *
 *
 *
 ************************************/
void MarkerDetector::drawContour(Mat &in, vector< Point > &contour, Scalar color) {
    for (unsigned int i = 0; i < contour.size(); i++) {
        cv::rectangle(in, contour[i], contour[i], color);
    }
}

void MarkerDetector::drawApproxCurve(Mat &in, vector< Point > &contour, Scalar color) {
    for (unsigned int i = 0; i < contour.size(); i++) {
        cv::line(in, contour[i], contour[(i + 1) % contour.size()], color);
    }
}
/************************************
 *
 *
 *
 *
 ************************************/

void MarkerDetector::draw(Mat out, const vector< Marker > &markers) {
    for (unsigned int i = 0; i < markers.size(); i++) {
        cv::line(out, markers[i][0], markers[i][1], cvScalar(255, 0, 0), 2, CV_AA);
        cv::line(out, markers[i][1], markers[i][2], cvScalar(255, 0, 0), 2, CV_AA);
        cv::line(out, markers[i][2], markers[i][3], cvScalar(255, 0, 0), 2, CV_AA);
        cv::line(out, markers[i][3], markers[i][0], cvScalar(255, 0, 0), 2, CV_AA);
    }
}
/* Attempt to make it faster than in opencv. I could not :( Maybe trying with SSE3...
void MarkerDetector::warpPerspective(const cv::Mat &in,cv::Mat & out, const cv::Mat & M,cv::Size size)
{
   //inverse the matrix
   out.create(size,in.type());
   //convert to float to speed up operations
   const double *m=M.ptr<double>(0);
   float mf[9];
   mf[0]=m[0];mf[1]=m[1];mf[2]=m[2];
   mf[3]=m[3];mf[4]=m[4];mf[5]=m[5];
   mf[6]=m[6];mf[7]=m[7];mf[8]=m[8];

   for(int y=0;y<out.rows;y++){
     uchar *_ptrout=out.ptr<uchar>(y);
     for(int x=0;x<out.cols;x++){
   //get the x,y position
   float den=1./(x*mf[6]+y*mf[7]+mf[8]);
   float ox= (x*mf[0]+y*mf[1]+mf[2])*den;
   float oy= (x*mf[3]+y*mf[4]+mf[5])*den;
   _ptrout[x]=in.at<uchar>(oy,ox);
     }
   }
}
*/



void MarkerDetector::findCornerMaxima(vector< cv::Point2f > &Corners, const cv::Mat &grey, int wsize) {

// for each element, search in a region around
#pragma omp parallel for

    for (size_t i = 0; i < Corners.size(); i++) {
        cv::Point2f minLimit(std::max(0, int(Corners[i].x - wsize)), std::max(0, int(Corners[i].y - wsize)));
        cv::Point2f maxLimit(std::min(grey.cols, int(Corners[i].x + wsize)), std::min(grey.rows, int(Corners[i].y + wsize)));

        cv::Mat reg = grey(cv::Range(minLimit.y, maxLimit.y), cv::Range(minLimit.x, maxLimit.x));
        cv::Mat harr, harrint;
        cv::cornerHarris(reg, harr, 3, 3, 0.04);

        // now, do a sum block operation
        cv::integral(harr, harrint);
        int bls_a = 4;
        for (int y = bls_a; y < harr.rows - bls_a; y++) {
            float *h = harr.ptr< float >(y);
            for (int x = bls_a; x < harr.cols - bls_a; x++)
                h[x] = harrint.at< double >(y + bls_a, x + bls_a) - harrint.at< double >(y + bls_a, x) - harrint.at< double >(y, x + bls_a) +
                       harrint.at< double >(y, x);
        }



        cv::Point2f best(-1, -1);
        cv::Point2f center(reg.cols / 2, reg.rows / 2);
        ;
        double maxv = 0;
        for (size_t i = 0; i < harr.rows; i++) {
            // L1 dist to center
            float *har = harr.ptr< float >(i);
            for (size_t x = 0; x < harr.cols; x++) {
                float d = float(fabs(center.x - x) + fabs(center.y - i)) / float(reg.cols / 2 + reg.rows / 2);
                float w = 1. - d;
                if (w * har[x] > maxv) {
                    maxv = w * har[x];
                    best = cv::Point2f(x, i);
                }
            }
        }
        Corners[i] = best + minLimit;
    }
}


void MarkerDetector::setMarkerLabeler(cv::Ptr<MarkerLabeler> detector)throw(cv::Exception){
    markerIdDetector=detector;
    if (markerIdDetector->getBestInputSize()!=-1)setWarpSize(markerIdDetector->getBestInputSize());

}

void MarkerDetector::setDictionary(Dictionary::DICT_TYPES dict_type,float error_correction_rate)throw(cv::Exception){
    markerIdDetector= MarkerLabeler::create(dict_type,error_correction_rate);
    if (markerIdDetector->getBestInputSize()!=-1)setWarpSize(markerIdDetector->getBestInputSize());
}
void MarkerDetector::setDictionary(string dict_type,float error_correction_rate)throw(cv::Exception){
    markerIdDetector= MarkerLabeler::create(Dictionary::getTypeFromString( dict_type),error_correction_rate);
    if (markerIdDetector->getBestInputSize()!=-1)setWarpSize(markerIdDetector->getBestInputSize());
}

};

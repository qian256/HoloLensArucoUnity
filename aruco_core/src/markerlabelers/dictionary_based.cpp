#include "dictionary_based.h"
#include <bitset>
#include <opencv2/imgproc/imgproc.hpp>


namespace aruco{

void DictionaryBased::setParams(const Dictionary &dic,float max_correction_rate){
    _dic=dic;
    max_correction_rate=max(0.f,min(1.0f,max_correction_rate));
    _maxCorrectionAllowed=float(_dic.tau())*max_correction_rate;

}

std::string DictionaryBased::getName()const{
    return aruco::Dictionary::getTypeString( _dic.getType());
}

void DictionaryBased::toMat(uint64_t code,int nbits_sq,cv::Mat  &out) {
    out.create(nbits_sq,nbits_sq,CV_8UC1);
    bitset<64> bs(code);
    int curbit=0;
    for(int r=0;r<nbits_sq;r++){
        uchar *pr=out.ptr<uchar>(r);
        for(int c=0;c<nbits_sq;c++)
              pr[c] = bs[curbit];

    }
}


int hamm_distance(uint64_t a,uint64_t b){
    uint64_t v=a&b;
    uint64_t mask=0x1;
    int d=0;
    for(int i=0;i<63;i++){
        d+= mask&v;
        v<<1;
    }
    return d;
}

bool DictionaryBased::detect(const cv::Mat &in, int & marker_id,int &nRotations) {
     assert(in.rows == in.cols);
    cv::Mat grey;
    if (in.type() == CV_8UC1) grey = in;
    else cv::cvtColor(in, grey, CV_BGR2GRAY);
    // threshold image
    cv::threshold(grey, grey, 125, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

     vector<uint64_t> ids;
    //get the ids in the four rotations (if possible)
    if ( !getInnerCode( grey,_dic.nbits(),ids)) return false;

     //find the best one
    for(int i=0;i<4;i++){
            if ( _dic.is(ids[i])){//is in the set?
                nRotations=i;//how many rotations are and its id
                marker_id=_dic[ids[i]];
                return true;//bye bye
            }
    }

    //you get here, no valid id :(
    //lets try error correction

    if(_maxCorrectionAllowed>0){//find distance to map elements
        for(auto ci:_dic.getMapCode()){
            for(int i=0;i<4;i++){
                if (hamm_distance(ci.first,ids[i])<_maxCorrectionAllowed){
                    marker_id=ci.second;
                    nRotations=i;
                    return true;
                }
            }
        }
    }
    else  return false;

 }

 bool DictionaryBased::getInnerCode(const cv::Mat &thres_img,int total_nbits,vector<uint64_t> &ids){
     int bits_a=sqrt(total_nbits);
    int bits_a2=bits_a+2;
    // Markers  are divided in (bits_a+2)x(bits_a+2) regions, of which the inner bits_axbits_a belongs to marker info
    // the external border shoould be entirely black

    int swidth = thres_img.rows / bits_a2;
    for (int y = 0; y < bits_a2; y++) {
        int inc = bits_a2-1;
        if (y == 0 || y == bits_a2-1)
            inc = 1; // for first and last row, check the whole border
        for (int x = 0; x < bits_a2; x += inc) {
            cv::Mat square = thres_img(cv::Rect(x*swidth, y*swidth, swidth, swidth));
            if (cv::countNonZero(square) > (swidth * swidth) / 2)
                return false; // can not be a marker because the border element is not black!
        }
    }

     // now,
    // get information(for each inner square, determine if it is  black or white)

    // now,
    cv::Mat _bits = cv::Mat::zeros(bits_a, bits_a, CV_8UC1);
    // get information(for each inner square, determine if it is  black or white)

    for (int y = 0; y < bits_a; y++) {

        for (int x = 0; x < bits_a; x++) {
            int Xstart = (x + 1) * (swidth);
            int Ystart = (y + 1) * (swidth);
            cv::Mat square = thres_img(cv::Rect(Xstart, Ystart, swidth, swidth));
            int nZ = cv::countNonZero(square);
            if (nZ > (swidth * swidth) / 2)
                _bits.at< uchar >(y, x) = 1;
        }
    }
    //now, get the 64bits ids

    int nr=0;
    do{
        ids.push_back(touulong(_bits));
        _bits=rotate(_bits);
        nr++;
    }while(nr<4);
     return true;
 }

 //convert matrix of (0,1)s in a 64 bit value
 uint64_t DictionaryBased::touulong(const cv::Mat &code){

     std::bitset<64> bits;
     int bidx=0;
     for (int y = code.rows-1; y >=0 ; y--)
         for (int x =  code.cols-1; x >=0; x--)
             bits[bidx++]=code.at<uchar>(y,x);
     return bits.to_ullong();

 }
 cv::Mat DictionaryBased::rotate(const cv::Mat &in) {
     cv::Mat out;
     in.copyTo(out);
     for (int i = 0; i < in.rows; i++) {
         for (int j = 0; j < in.cols; j++) {
             out.at< uchar >(i, j) = in.at< uchar >(in.cols - j - 1, i);
         }
     }
     return out;
 }


}

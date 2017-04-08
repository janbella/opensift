#ifndef SIFTFEATURE_H
#define SIFTFEATURE_H

/**
 * @file siftfeature.h
 * @author Ján Bella
 **/


#include <opencv2/opencv.hpp>

#define FEATURE_MAX_D 128

namespace Stitching
{

class SiftFeature
{
public:
    double x;                      // x coord
    double y;                      // y coord
    double scl;                    // scale of a Lowe-style feature
    double ori;                    // orientation of a Lowe-style feature
    unsigned short d;              // descriptor length
    double descr[FEATURE_MAX_D];   // descriptor
    cv::Point2d img_pt;            // location in image


    unsigned short ddata_r;
    unsigned short ddata_c;

    unsigned short ddata_intvl;
    double ddata_subintvl;

    unsigned short ddata_octv;
    double ddata_scl_octv;



    SiftFeature();

    SiftFeature(const SiftFeature& other);

    ~SiftFeature();

};
}
#endif // SIFTFEATURE_H

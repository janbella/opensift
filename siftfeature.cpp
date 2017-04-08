/**
 * @file siftfeature.cpp
 * @author Ján Bella
 **/


#include "siftfeature.h"

using namespace Stitching;


SiftFeature::SiftFeature(): x(0), y(0), scl(0), ori(0), d(0), ddata_r(0),
    ddata_c(0), ddata_intvl(0), ddata_subintvl(0), ddata_octv(0), ddata_scl_octv(0)
{
    memset( descr, 0, FEATURE_MAX_D * sizeof( double ) );
}

SiftFeature::SiftFeature(const SiftFeature& other): x(other.x), y(other.y), scl(other.scl), ori(other.ori),
    d(other.d), img_pt(other.img_pt), ddata_r(other.ddata_r), ddata_c(other.ddata_c), ddata_intvl(other.ddata_intvl),
    ddata_subintvl(other.ddata_subintvl), ddata_octv(other.ddata_octv), ddata_scl_octv(other.ddata_scl_octv)
{
    std::copy(other.descr, other.descr + other.d, descr);
}

SiftFeature::~SiftFeature()
{

}

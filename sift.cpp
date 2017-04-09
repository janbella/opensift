/**
 * @file sift.cpp
 * @author Ján Bella <xbella1@fi.muni.cz>
 *
 * The file provides the definition of SIFT class, which is an object for computing SIFT keypoints and descriptors
 *
 * The implementation is based on Rob Hess's opensift, see the original copyright below.
 */

/**
 * Functions for detecting SIFT image features.
 *
 * For more information, refer to:
 *
 * Lowe, D.  Distinctive image features from scale-invariant keypoints.
 * <EM>International Journal of Computer Vision, 60</EM>, 2 (2004), pp.91--110.
 *
 * Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>
 *
 * Note: The SIFT algorithm is patented in the United States and cannot be used in commercial products
 * without a license from the University of British Columbia.  For more information, refer to the file
 * LICENSE.ubc that accompanied this distribution.
 *
 * @version 1.1.2-20100521
 **/

#include "sift.h"
#include "siftfeature.h"

using namespace Stitching;


// absolute value
#ifndef ABS
#define ABS(x) ( ( (x) < 0 )? -(x) : (x) )
#endif

SIFT::SIFT(): octvs(0),
    intvls(SIFT_INTVLS), sigma(SIFT_SIGMA), contr_thr(SIFT_CONTR_THR), curv_thr(SIFT_CURV_THR), img_dbl(SIFT_IMG_DBL),
    descr_width(SIFT_DESCR_WIDTH), descr_hist_bins(SIFT_DESCR_HIST_BINS)
{
    dog_pyr = nullptr;
    gauss_pyr = nullptr;
    gaussPyrIncrSig = new double[intvls + 3];
    precompute_incremental_sigmas();
    orientationHistogram = new double[SIFT_ORI_HIST_BINS];
    alloc_descr_hist();
}

SIFT::SIFT(unsigned short intvls, double sigma, double contr_thr, unsigned short curv_thr,
     bool img_dbl, unsigned short descr_width, unsigned short descr_hist_bins):
     octvs(0),intvls(intvls), sigma(sigma), contr_thr(contr_thr), curv_thr(curv_thr), img_dbl(img_dbl),
    descr_width(descr_width), descr_hist_bins(descr_hist_bins)
{
    dog_pyr = nullptr;
    gauss_pyr = nullptr;
    gaussPyrIncrSig = new double[intvls + 3];
    precompute_incremental_sigmas();
    orientationHistogram = new double[SIFT_ORI_HIST_BINS];
    alloc_descr_hist();
}

SIFT::SIFT(const SIFT& other):
    intvls(other.intvls), sigma(other.sigma), contr_thr(other.contr_thr), curv_thr(other.curv_thr), img_dbl(other.img_dbl),
    descr_width(other.descr_width), descr_hist_bins(other.descr_hist_bins)
{
    if(octvs > 0)
    {
        alloc_pyr(gauss_pyr, intvls + 3 );
        alloc_pyr(dog_pyr, intvls + 2 );

        for (unsigned short  i = 0; i < octvs; i++ )
        {
            for (unsigned short  j = 0; j < intvls + 3; j++ )
            {
                gauss_pyr[i][j] = other.gauss_pyr[i][j].clone();
            }

            for (unsigned short  j = 0; j < intvls + 2; j++ )
            {
                dog_pyr[i][j] = other.dog_pyr[i][j].clone();
            }
        }
    }

    gaussPyrIncrSig = new double[intvls + 3];
    precompute_incremental_sigmas();
    orientationHistogram = new double[SIFT_ORI_HIST_BINS];
    alloc_descr_hist();
}


SIFT::~SIFT()
{
    release_pyr( gauss_pyr, intvls + 3 );
    release_pyr( dog_pyr, intvls + 2 );
    release_descr_hist();


    delete[] gaussPyrIncrSig;
    delete[] orientationHistogram;

}

unsigned short SIFT::recommendedOctaves(unsigned short rows, unsigned short cols)
{
    return static_cast<unsigned short>(log( MIN( cols, rows) ) / log(2) - 2);
}

unsigned short SIFT::recommendedOctaves(const cv::Mat& image)
{
    return static_cast<unsigned short>(log( MIN( image.cols, image.rows) ) / log(2) - 2);
}

std::vector<SiftFeature*> SIFT::extractFeatures(cv::Mat &img)
{
    if ( img.empty() )
    {
        std::cerr << "Input image is empty." << std::endl;
        return std::vector<SiftFeature*>();
    }

    // build scale space pyramid; smallest dimension of top level is ~4 pixels
    cv::Mat init_img = create_init_img( img );

    if(octvs == 0)
    {
        octvs = recommendedOctaves(init_img);
    }

    if(gauss_pyr == nullptr || dog_pyr == nullptr)
    {
        alloc_pyr(gauss_pyr, intvls + 3 );
        alloc_pyr(dog_pyr, intvls + 2 );
    }

    build_gauss_pyr( init_img );
    build_dog_pyr();

    std::vector<SiftFeature*> features = scale_space_extrema();
    calc_feature_scales( features);

    if ( img_dbl )
    {
        adjust_for_img_dbl( features );
    }
    calc_feature_oris( features );
    compute_descriptors( features );

    return features;
}



cv::Mat  SIFT::create_init_img( cv::Mat const& img )
{
    cv::Mat gray = convert_to_gray32( img );

    if ( img_dbl )
    {
        double sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4 );
        cv::Mat  dbl = cv::Mat(img.rows * 2, img.cols * 2, CV_32FC1);
        cv::resize(gray,dbl,dbl.size(),0,0,cv::INTER_CUBIC);
        cv::GaussianBlur(dbl,dbl,cv::Size(),sig_diff,sig_diff,cv::BORDER_REPLICATE);
        return dbl;
    }
    else
    {
        double sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA );
        cv::GaussianBlur(gray,gray,cv::Size(),sig_diff,sig_diff,cv::BORDER_REPLICATE);
        return gray;
    }
}


cv::Mat SIFT::convert_to_gray32( cv::Mat const& img )
{
    cv::Mat gray8;

    if ( img.channels() == 1 )
    {
        gray8 = img.clone();
    }
    else
    {
        cv::cvtColor(img, gray8, CV_BGR2GRAY );
    }

    cv::Mat gray32;
    gray8.convertTo(gray32, CV_32FC1);
    gray32 = gray32 / 255.0;

    return gray32;
}


void SIFT::build_gauss_pyr(  cv::Mat const& base )
{
    for (unsigned short o = 0; o < octvs; o++ )
    {
        if(o == 0)
        {
            gauss_pyr[o][0] = base.clone();
        }
        // base of new octvave is halved image from end of previous octave
        else
        {
            // downsample to quarter size (half in each dimension) using nearest-neighbor interpolation
            cv::resize(gauss_pyr[o - 1][intvls], gauss_pyr[o][0], cv::Size(gauss_pyr[o - 1][intvls].cols / 2,
                    gauss_pyr[o - 1][intvls].rows / 2), 0, 0, cv::INTER_NEAREST);
        }

        for (unsigned short i = 1; i < intvls + 3; i++ )
        {
            // blur the current octave's last image to create the next one
            cv::GaussianBlur(gauss_pyr[o][i - 1], gauss_pyr[o][i], cv::Size(), gaussPyrIncrSig[i], gaussPyrIncrSig[i], cv::BORDER_REPLICATE );
        }
    }
}


void SIFT::build_dog_pyr()
{
    for (unsigned short o = 0; o < octvs; o++ )
    {
        for (unsigned short i = 0; i < intvls + 2; i++ )
        {
            dog_pyr[o][i] = cv::Mat(gauss_pyr[o][i].rows,gauss_pyr[o][i].cols, CV_32FC1);
            cv::subtract(gauss_pyr[o][i + 1], gauss_pyr[o][i], dog_pyr[o][i] );
        }
    }
}


std::vector<SiftFeature*> SIFT::scale_space_extrema()
{
    std::vector<SiftFeature*>  features;
    double prelim_contr_thr = 0.5 * contr_thr / intvls;


    for (unsigned short  o = 0; o < octvs; o++ )
    {
        for (unsigned short  i = 1; i <= intvls; i++ )
        {
            for (unsigned short r = SIFT_IMG_BORDER; r < dog_pyr[o][0].rows - SIFT_IMG_BORDER; r++)
            {
                for (unsigned short c = SIFT_IMG_BORDER; c < dog_pyr[o][0].cols - SIFT_IMG_BORDER; c++)
                {
                    // perform preliminary check on
                    if ( ABS( dog_pyr[o][i].at<float>(r, c) ) > prelim_contr_thr )
                    {
                        if ( is_extremum(dog_pyr, o, i, r, c ) )
                        {
                            SiftFeature* feat = interp_extremum(dog_pyr, o, i, r, c);
                            if ( feat )
                            {
                                if ( ! is_too_edge_like( dog_pyr[feat->ddata_octv][feat->ddata_intvl],
                                                         feat->ddata_r, feat->ddata_c ) )
                                {
                                    features.push_back(feat);
                                }
                                else
                                {
                                    delete feat;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return features;
}


bool SIFT::is_extremum(cv::Mat** const& dog_pyr,  unsigned short octv, unsigned short intvl,
                         unsigned short r, unsigned short c )
{
    double val = dog_pyr[octv][intvl].at<float>( r, c );

    // check for maximum
    if ( val > 0 )
    {
        for (short i = -1; i <= 1; i++ )
        {
            for (short j = -1; j <= 1; j++ )
            {
                for (short k = -1; k <= 1; k++ )
                {
                    if ( val < dog_pyr[octv][intvl + i].at<float>( r + j, c + k ) )
                    {
                        return false;
                    }
                }
            }
        }
    }

    // check for minimum
    else
    {
        for (short i = -1; i <= 1; i++ )
        {
            for (short j = -1; j <= 1; j++ )
            {
                for (short k = -1; k <= 1; k++ )
                {
                    if ( val > dog_pyr[octv][intvl + i].at<float>( r + j, c + k ) )
                    {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}


SiftFeature* SIFT::interp_extremum(cv::Mat** const& dog_pyr,  unsigned short octv,  unsigned short intvl,
                                     unsigned short r, unsigned short c )
{
    double xi, xr, xc, contr;
    unsigned short i = 0;

    while ( i < SIFT_MAX_INTERP_STEPS )
    {
        interp_step( dog_pyr, octv, intvl, r, c, &xi, &xr, &xc );
        if ( ABS( xi ) < 0.5  &&  ABS( xr ) < 0.5  &&  ABS( xc ) < 0.5 )
        {
            break;
        }

        c += cvRound( xc );
        r += cvRound( xr );
        intvl += cvRound( xi );

        if ( intvl < 1  || intvl > intvls  ||
             c < SIFT_IMG_BORDER  || r < SIFT_IMG_BORDER  ||
             c >= dog_pyr[octv][0].cols - SIFT_IMG_BORDER  || r >= dog_pyr[octv][0].rows - SIFT_IMG_BORDER )
        {
            return nullptr;
        }
        i++;
    }

    // ensure convergence of interpolation
    if ( i >= SIFT_MAX_INTERP_STEPS )
    {
        return nullptr;
    }

    contr = interp_contr( dog_pyr, octv, intvl, r, c, xi, xr, xc );
    if ( ABS( contr ) < contr_thr / intvls )
    {
        return nullptr;
    }

    SiftFeature* feat = new SiftFeature();
    feat->img_pt.x = feat->x = ( c + xc ) * pow( 2.0, octv );
    feat->img_pt.y = feat->y = ( r + xr ) * pow( 2.0, octv );
    feat->ddata_r = r;
    feat->ddata_c = c;
    feat->ddata_octv = octv;
    feat->ddata_intvl = intvl;
    feat->ddata_subintvl = xi;

    return feat;
}


void SIFT::interp_step(cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl, unsigned short r,
                         unsigned short c, double* xi, double* xr, double* xc )
{
    cv::Matx31d X(0,0,0);

    cv::Matx31d dD = deriv_3D( dog_pyr, octv, intvl, r, c );
    cv::Matx33d H = hessian_3D( dog_pyr, octv, intvl, r, c );
    cv::Matx33d H_inv;
    cv::invert(H,H_inv, cv::DECOMP_SVD);
    cv::gemm(H_inv, dD, -1, cv::Matx<double,0,0>(), 0, X, 0);

    *xi = X(2);
    *xr = X(1);
    *xc = X(0);
}


cv::Matx31d SIFT::deriv_3D(cv::Mat** const& dog_pyr,  unsigned short octv, unsigned short intvl,
                             unsigned short r, unsigned short c )
{
    double dx = ( dog_pyr[octv][intvl].at<float>( r, c + 1 )
                  - dog_pyr[octv][intvl].at<float>( r, c - 1 ) ) / 2.0;
    double dy = (  dog_pyr[octv][intvl].at<float>( r + 1, c )
                   -  dog_pyr[octv][intvl].at<float>( r - 1, c ) ) / 2.0;
    double ds = ( dog_pyr[octv][intvl + 1].at<float>( r, c )
                - dog_pyr[octv][intvl - 1].at<float>( r, c ) ) / 2.0;

    return cv::Matx31d( dx,dy,ds);
}


cv::Matx33d SIFT::hessian_3D(cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl,
                               unsigned short r, unsigned short c )
{
    double v =  dog_pyr[octv][intvl].at<float>( r, c );
    double dxx = ( dog_pyr[octv][intvl].at<float>( r, c + 1 ) +
                   dog_pyr[octv][intvl].at<float>( r, c - 1 ) - 2 * v );
    double dyy = ( dog_pyr[octv][intvl].at<float>( r + 1, c ) +
                   dog_pyr[octv][intvl].at<float>( r - 1, c ) - 2 * v );
    double dss = ( dog_pyr[octv][intvl + 1].at<float>( r, c ) +
                 dog_pyr[octv][intvl - 1].at<float>( r, c ) - 2 * v );
    double dxy = ( dog_pyr[octv][intvl].at<float>( r + 1, c + 1 ) -
                   dog_pyr[octv][intvl].at<float>( r + 1, c - 1 ) -
                   dog_pyr[octv][intvl].at<float>( r - 1, c + 1 ) +
                   dog_pyr[octv][intvl].at<float>( r - 1, c - 1 ) ) / 4.0;
    double dxs = ( dog_pyr[octv][intvl + 1].at<float>( r, c + 1 ) -
                 dog_pyr[octv][intvl + 1].at<float>( r, c - 1 ) -
                 dog_pyr[octv][intvl - 1].at<float>( r, c + 1 ) +
                 dog_pyr[octv][intvl - 1].at<float>( r, c - 1 ) ) / 4.0;
    double dys = ( dog_pyr[octv][intvl + 1].at<float>( r + 1, c ) -
                 dog_pyr[octv][intvl + 1].at<float>( r - 1, c ) -
                 dog_pyr[octv][intvl - 1].at<float>( r + 1, c ) +
                 dog_pyr[octv][intvl - 1].at<float>( r - 1, c ) ) / 4.0;

    return cv::Matx33d(dxx,dxy,dxs,
                       dxy,dyy,dys,
                       dxs,dys,dss);
}


double SIFT::interp_contr(cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl,
                            unsigned short r, unsigned short c, double xi, double xr, double xc )
{
    cv::Matx31d X(xc,xr,xi);
    cv::Matx<double,1,1> T;
    cv::Matx31d dD = deriv_3D( dog_pyr, octv, intvl, r, c );

    cv::gemm( dD, X, 1, cv::Matx<double,0,0>(), 0, T,  cv::GEMM_1_T );

    return dog_pyr[octv][intvl].at<float>( r, c ) + T(0) * 0.5;
}


bool SIFT::is_too_edge_like( const cv::Mat& dog_img, unsigned short r, unsigned short c )
{
    // principal curvatures are computed using the trace and det of Hessian
    double d = dog_img.at<float>(r, c);
    double dxx = dog_img.at<float>( r, c + 1 ) +  dog_img.at<float>( r, c - 1 ) - 2 * d;
    double dyy =  dog_img.at<float>( r + 1, c ) +  dog_img.at<float>( r - 1, c ) - 2 * d;
    double dxy = ( dog_img.at<float>( r + 1, c + 1) - dog_img.at<float>( r + 1, c - 1) -
                   dog_img.at<float>( r - 1, c + 1) + dog_img.at<float>( r - 1, c - 1) ) / 4.0;
    double tr = dxx + dyy;
    double det = dxx * dyy - dxy * dxy;

    // negative determinant -> curvatures have different signs; reject feature
    if ( det <= 0 )
    {
        return true;
    }

    if ( tr * tr / det < ( curv_thr + 1.0 ) * ( curv_thr + 1.0 ) / curv_thr )
    {
        return false;
    }
    return true;
}


void SIFT::calc_feature_scales( std::vector<SiftFeature*>& features )
{
    for (unsigned short i = 0; i < features.size(); i++ )
    {
        SiftFeature* feat = features[i];
        double intvl = feat->ddata_intvl + feat->ddata_subintvl;
        feat->scl = sigma * pow( 2.0, feat->ddata_octv + intvl / intvls );
        feat->ddata_scl_octv = sigma * pow( 2.0, intvl / intvls );
    }
}


void SIFT::adjust_for_img_dbl( std::vector<SiftFeature*>& features )
{
    for (unsigned short  i = 0; i < features.size(); i++ )
    {
        SiftFeature* feat = features[i];
        feat->x /= 2.0;
        feat->y /= 2.0;
        feat->scl /= 2.0;
        feat->img_pt.x /= 2.0;
        feat->img_pt.y /= 2.0;
    }
}


void SIFT::calc_feature_oris( std::vector<SiftFeature*>& features )
{
    unsigned short n = features.size();

    for (unsigned short i = 0; i < n; i++ )
    {
        SiftFeature* feat =  features[i];
        compute_ori_hist( gauss_pyr[feat->ddata_octv][feat->ddata_intvl],
                feat->ddata_r, feat->ddata_c,
                cvRound( SIFT_ORI_RADIUS * feat->ddata_scl_octv ),
                SIFT_ORI_SIG_FCTR * feat->ddata_scl_octv );
        for (unsigned short j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++ )
        {
            smooth_ori_hist();
        }
        double omax = dominant_ori();
        add_good_ori_features( features, omax * SIFT_ORI_PEAK_RATIO, feat );
        delete feat;
    }
    features.erase(features.begin(), features.begin() + n);
}

void SIFT::compute_ori_hist( const cv::Mat& img, unsigned short r, unsigned short c, int rad, double sigma )
{
    double mag, ori, PI2 = CV_PI * 2.0;

    memset(orientationHistogram,0,SIFT_ORI_HIST_BINS * sizeof(double));

    double exp_denom = 2.0 * sigma * sigma;
    for (int  i = -rad; i <= rad; i++ )
    {
        for (int  j = -rad; j <= rad; j++ )
        {
            if ( calc_grad_mag_ori( img, r + i, c + j, &mag, &ori ) )
            {
                double w = exp( -( i * i + j * j ) / exp_denom );
                int bin = cvRound( SIFT_ORI_HIST_BINS * ( ori + CV_PI ) / PI2 );
                bin = ( bin < SIFT_ORI_HIST_BINS ) ? bin : 0;
                orientationHistogram[bin] += w * mag;
            }
        }
    }
}


bool SIFT::calc_grad_mag_ori( const cv::Mat& img, unsigned short r, unsigned short c, double* mag, double* ori )
{
    if ( r > 0  &&  r < img.rows - 1  &&  c > 0  &&  c < img.cols - 1 )
    {
        double dx = img.at<float>(r, c + 1 ) - img.at<float>( r, c - 1 );
        double dy = img.at<float>( r - 1, c ) - img.at<float>( r + 1, c );
        *mag = sqrt( dx * dx + dy * dy );
        *ori = atan2( dy, dx );
        return true;
    }
    else
    {
        return false;
    }
}


void SIFT::smooth_ori_hist( )
{
    double h0 = orientationHistogram[0];

    double prev = orientationHistogram[SIFT_ORI_HIST_BINS - 1];
    for (unsigned short i = 0; i < SIFT_ORI_HIST_BINS; i++ )
    {
        double tmp = orientationHistogram[i];
        orientationHistogram[i] = 0.25 * prev + 0.5 * orientationHistogram[i] +
                  0.25 * ( ( i + 1 == SIFT_ORI_HIST_BINS ) ? h0 : orientationHistogram[i + 1] );
        prev = tmp;
    }
}


double SIFT::dominant_ori()
{
    double omax = orientationHistogram[0];
    //int maxbin = 0;
    for (unsigned short i = 1; i < SIFT_ORI_HIST_BINS; i++ )
    {
        if ( orientationHistogram[i] > omax )
        {
            omax = orientationHistogram[i];
            //maxbin = i;
        }
    }
    return omax;
}


void SIFT::add_good_ori_features( std::vector<SiftFeature*>& features, double mag_thr, SiftFeature* feat )
{
    double PI2 = CV_PI * 2.0;

    for (unsigned short i = 0; i < SIFT_ORI_HIST_BINS; i++ )
    {
        int l = ( i == 0 ) ? SIFT_ORI_HIST_BINS - 1 : i - 1;
        int r = ( i + 1 ) % SIFT_ORI_HIST_BINS;

        if ( orientationHistogram[i] > orientationHistogram[l]  &&  orientationHistogram[i] > orientationHistogram[r]  &&  orientationHistogram[i] >= mag_thr )
        {
            double bin = i + interp_hist_peak( orientationHistogram[l], orientationHistogram[i], orientationHistogram[r] );
            bin = ( bin < 0 ) ? SIFT_ORI_HIST_BINS + bin : ( bin >= SIFT_ORI_HIST_BINS ) ? bin - SIFT_ORI_HIST_BINS : bin;
            SiftFeature* new_feat = new SiftFeature( *feat );
            new_feat->ori = ( ( PI2 * bin ) / SIFT_ORI_HIST_BINS ) - CV_PI;
            features.push_back(new_feat);
        }
    }
}


void SIFT::compute_descriptors( std::vector<SiftFeature*>& features)
{
    for (unsigned short i = 0; i < features.size(); i++ )
    {
        SiftFeature* feat = features[i];
        descr_hist( gauss_pyr[feat->ddata_octv][feat->ddata_intvl], feat->ddata_r,
                    feat->ddata_c, feat->ori, feat->ddata_scl_octv);
        hist_to_descr( feat );
    }
}


void SIFT::descr_hist( const cv::Mat& img, unsigned short r, unsigned short c, double ori, double scl)
{
    double grad_mag, grad_ori, PI2 = 2.0 * CV_PI;

    for (unsigned short i = 0; i < descr_width; i++ )
    {
        for (int j = 0; j < descr_width; j++ )
        {
            memset(descriptorHistogram[i][j],0,descr_hist_bins*sizeof(double));
        }
    }

    double cos_t = cos( ori );
    double sin_t = sin( ori );
    double bins_per_rad =descr_hist_bins/ PI2;
    double exp_denom =descr_width*descr_width* 0.5;
    double hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = hist_width * sqrt(2) * (descr_width+ 1.0 ) * 0.5 + 0.5;
    for (int  i = -radius; i <= radius; i++ )
    {
        for (int j = -radius; j <= radius; j++ )
        {
            /* Calculate sample's histogram array coords rotated relative to ori.
               Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
               r_rot = 1.5) have full weight placed in row 1 after interpolation. */

            double c_rot = ( j * cos_t - i * sin_t ) / hist_width;
            double r_rot = ( j * sin_t + i * cos_t ) / hist_width;
            double rbin = r_rot +descr_width/ 2 - 0.5;
            double cbin = c_rot +descr_width/ 2 - 0.5;

            if ( rbin > -1.0  &&  rbin <descr_width &&  cbin > -1.0  &&  cbin <descr_width)
            {
                if ( calc_grad_mag_ori( img, r + i, c + j, &grad_mag, &grad_ori ))
                {
                    grad_ori -= ori;
                    while ( grad_ori < 0.0 )
                    {
                        grad_ori += PI2;
                    }
                    while ( grad_ori >= PI2 )
                    {
                        grad_ori -= PI2;
                    }

                    double obin = grad_ori * bins_per_rad;
                    double w = exp( -(c_rot * c_rot + r_rot * r_rot) / exp_denom );
                    interp_desc_hist_entry(rbin, cbin, obin, grad_mag * w);
                }
            }
        }
    }
}


void SIFT::interp_desc_hist_entry( double rbin, double cbin,double obin, double mag)
{
    int r0 = cvFloor( rbin );
    int c0 = cvFloor( cbin );
    int o0 = cvFloor( obin );
    double d_r = rbin - r0;
    double d_c = cbin - c0;
    double d_o = obin - o0;

    /* The entry is distributed into up to 8 bins.  Each entry into a bin
       is multiplied by a weight of 1 - d for each dimension, where d is the
       distance from the center value of the bin measured in bin units.  */

    for (unsigned short r = 0; r <= 1; r++ )
    {
        int rb = r0 + r;
        if ( rb >= 0  &&  rb <descr_width)
        {
            double v_r = mag * ( ( r == 0 ) ? 1.0 - d_r : d_r );
            double** row = descriptorHistogram[rb];
            for (int c = 0; c <= 1; c++ )
            {
                int cb = c0 + c;
                if ( cb >= 0  &&  cb <descr_width)
                {
                    double v_c = v_r * ( ( c == 0 ) ? 1.0 - d_c : d_c );
                    double* h = row[cb];
                    for (int o = 0; o <= 1; o++ )
                    {
                        int ob = ( o0 + o ) % descr_hist_bins;
                        double v_o = v_c * ( ( o == 0 ) ? 1.0 - d_o : d_o );
                        h[ob] += v_o;
                    }
                }
            }
        }
    }
}


void SIFT::hist_to_descr( SiftFeature* feat )
{
    unsigned short k = 0;

    for (unsigned short  r = 0; r < descr_width; r++ )
    {
        for (unsigned short  c = 0; c < descr_width; c++ )
        {
            for (unsigned short  o = 0; o < descr_hist_bins; o++ )
            {
                feat->descr[k++] = descriptorHistogram[r][c][o];
            }
        }
    }

    feat->d = k;
    normalize_descr( feat );
    for (unsigned short i = 0; i < k; i++ )
    {
        if ( feat->descr[i] > SIFT_DESCR_MAG_THR )
        {
            feat->descr[i] = SIFT_DESCR_MAG_THR;
        }
    }
    normalize_descr( feat );

    // convert floating-point descriptor to integer valued descriptor
    for (unsigned short i = 0; i < k; i++ )
    {
        int int_val = SIFT_INT_DESCR_FCTR * feat->descr[i];
        feat->descr[i] = MIN( 255, int_val );
    }
}


void SIFT::normalize_descr( SiftFeature* feat )
{
    double len_sq = 0.0;
    int d = feat->d;

    for (int  i = 0; i < d; i++ )
    {
        double cur = feat->descr[i];
        len_sq += cur * cur;
    }
    double len_inv = 1.0 / sqrt( len_sq );
    for (int i = 0; i < d; i++ )
    {
        feat->descr[i] *= len_inv;
    }
}


void SIFT::release_descr_hist()
{
    for (unsigned short  i = 0; i < descr_width; i++)
    {
        for (unsigned short  j = 0; j < descr_width; j++ )
        {
            delete[] descriptorHistogram[i][j];
        }
        delete[] descriptorHistogram[i];
    }
    delete[] descriptorHistogram;
    descriptorHistogram = nullptr;
}


void SIFT::release_pyr( cv::Mat**& pyr, unsigned short n )
{
    for (unsigned short  i = 0; i < octvs; i++ )
    {
        for (unsigned short  j = 0; j < n; j++ )
        {
            pyr[i][j].release();
        }
        delete[] pyr[i];
    }
    delete[] pyr;
    pyr = nullptr;
}

void SIFT::alloc_pyr( cv::Mat**& pyr, unsigned short n )
{
    pyr = new cv::Mat*[octvs];
    for (unsigned short  i = 0; i < octvs; i++ )
    {
        pyr[i] = new cv::Mat[n] ;
    }
}

void SIFT::precompute_incremental_sigmas()
{
    double k = pow( 2.0, 1.0 / intvls );
    gaussPyrIncrSig[0] = sigma;
    gaussPyrIncrSig[1] = sigma * sqrt( k * k - 1 );
    for (unsigned short i = 2; i < intvls + 3; i++)
    {
        gaussPyrIncrSig[i] = gaussPyrIncrSig[i - 1] * k;
    }
}

void SIFT::alloc_descr_hist()
{
    descriptorHistogram = new double**[descr_width];
    for (unsigned short i = 0; i < descr_width; i++ )
    {
        descriptorHistogram[i] = new double*[descr_width];
        for (int j = 0; j < descr_width; j++ )
        {
            descriptorHistogram[i][j] = new double[descr_hist_bins];
            memset(descriptorHistogram[i][j],0,descr_hist_bins*sizeof(double));
        }
    }
}

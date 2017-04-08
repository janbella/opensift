/**
 * @file sift_c.cpp
 * @author Rob Hess, edited by Ján Bella.
 *
 * See original copyright below.
 **/


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

#include "sift_c.h"
#include "siftfeature.h"

#include <opencv2/opencv.hpp>

using namespace Stitching;


/************************* Local Function Prototypes *************************/

/**
 * Converts an image to 8-bit grayscale and Gaussian-smooths it. The image is
 * optionally doubled in size prior to smoothing.
 *
 * @param img input image
 * @param img_dbl if true, image is doubled in size prior to smoothing
 * @param sigma total std of Gaussian smoothing
 */
static cv::Mat  create_init_img( cv::Mat const& img, bool img_dbl, double sigma );

/**
 * Converts an image to 32-bit grayscale
 *
 * @param img a 3-channel 8-bit color (BGR) or 8-bit gray image
 * @return 32-bit grayscale image
 */
static cv::Mat convert_to_gray32( cv::Mat const& img );

/**
 * Builds Gaussian scale space pyramid from an image
 *
 * @param base base image of the pyramid
 * @param octvs number of octaves of scale space
 * @param intvls number of intervals per octave
 * @param sigma amount of Gaussian smoothing per octave
 * @return Gaussian scale space pyramid as an octvs x (intvls + 3) array
 */
static cv::Mat** build_gauss_pyr(  cv::Mat const& base, unsigned short octvs, unsigned short intvls, double sigma );

/**
 * Builds a difference of Gaussians scale space pyramid by subtracting adjacent intervals of a Gaussian pyramid
 *
 * @param gauss_pyr Gaussian scale-space pyramid
 * @param octvs number of octaves of scale space
 * @param intvls number of intervals per octave
 * @return Returns a difference of Gaussians scale space pyramid as an octvs x (intvls + 2) array
 */
static cv::Mat** build_dog_pyr( cv::Mat** const& gauss_pyr, unsigned short octvs, unsigned short intvls );

/**
 * Detects features at extrema in DoG scale space.  Bad features are discarded based on contrast
 * and ratio of principal curvatures.
 *
 * @param dog_pyr DoG scale space pyramid
 * @param octvs octaves of scale space represented by dog_pyr
 * @param intvls intervals per octave
 * @param contr_thr low threshold on feature contrast
 * @param curv_thr high threshold on feature ratio of principal curvatures
 * @param storage memory storage in which to store detected features
 * @return Returns an array of detected features whose scales, orientations, and descriptors are yet to be determined.
 */
static std::vector<SiftFeature*> scale_space_extrema( cv::Mat** const& dog_pyr, int octvs, int intvls,
                                                      double contr_thr, int curv_thr);

/**
 * Determines whether a pixel is a scale-space extremum by comparing it to it's 3x3x3 pixel neighborhood.
 *
 * @param dog_pyr DoG scale space pyramid
 * @param octv pixel's scale space octave
 * @param intvl pixel's within-octave interval
 * @param r pixel's image row
 * @param c pixel's image col
 * @return true if the specified pixel is an extremum (max or min) among it's 3x3x3 pixel neighborhood.
 */
static bool is_extremum( cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl,
                         unsigned short r, unsigned short c );

/**
 * Interpolates a scale-space extremum's location and scale to subpixel accuracy to form an image feature.
 * Rejects features with low contrast. Based on Section 4 of Lowe's paper.
 *
 * @param dog_pyr DoG scale space pyramid
 * @param octv feature's octave of scale space
 * @param intvl feature's within-octave interval
 * @param r feature's image row
 * @param c feature's image column
 * @param intvls total intervals per octave
 * @param contr_thr threshold on feature contrast
 *
 * @return Returns the feature resulting from interpolation of the given parameters or nullptr if the given
 *   location could not be interpolated or if contrast at the interpolated loation was too low.  If a feature
 *   is returned, its scale, orientation, and descriptor are yet to be determined.
 */
static SiftFeature* interp_extremum( cv::Mat** const& dog_pyr, unsigned short octv,  unsigned short intvl,
                                     unsigned short r, unsigned short c, unsigned short intvls, double contr_thr );

/**
 * Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's paper.
 *
 * @param dog_pyr difference of Gaussians scale space pyramid
 * @param octv octave of scale space
 * @param intvl interval being interpolated
 * @param r row being interpolated
 * @param c column being interpolated
 * @param xi output as interpolated subpixel increment to interval
 * @param xr output as interpolated subpixel increment to row
 * @param xc output as interpolated subpixel increment to col
 */

static void interp_step( cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl, unsigned short r,
                         unsigned short c, double* xi, double* xr, double* xc );

/**
 * Computes the partial derivatives in x, y, and scale of a pixel in the DoG scale space pyramid.
 *
 * @param dog_pyr DoG scale space pyramid
 * @param octv pixel's octave in dog_pyr
 * @param intvl pixel's interval in octv
 * @param r pixel's image row
 * @param c pixel's image col
 * @return vector of partial derivatives for pixel I { dI/dx, dI/dy, dI/ds }^T as a cv::Matx31d
 */
static cv::Matx31d deriv_3D( cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl,
                             unsigned short r, unsigned short c );

/**
 * Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.
 *
 * @param dog_pyr DoG scale space pyramid
 * @param octv pixel's octave in dog_pyr
 * @param intvl pixel's interval in octv
 * @param r pixel's image row
 * @param c pixel's image col
 *
 * @return the Hessian matrix (below) for pixel I as a cv::Matx33d
 *
 * / Ixx  Ixy  Ixs \ <BR>
 * | Ixy  Iyy  Iys | <BR>
 * \ Ixs  Iys  Iss /
 */
static cv::Matx33d hessian_3D( cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl,
                               unsigned short r, unsigned short c );

/**
 * Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's paper.
 *
 * @param dog_pyr difference of Gaussians scale space pyramid
 * @param octv octave of scale space
 * @param intvl within-octave interval
 * @param r pixel row
 * @param c pixel column
 * @param xi interpolated subpixel increment to interval
 * @param xr interpolated subpixel increment to row
 * @param xc interpolated subpixel increment to col
 *
 * @return interpolated contrast
 */
static double interp_contr( cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl,
                            unsigned short r, unsigned short c, double xi, double xr, double xc );

/**
 * Determines whether a feature is too edge like to be stable by computing the ratio of principal
 *    curvatures at that feature.  Based on Section 4.1 of Lowe's paper.
 *
 * @param dog_img image from the DoG pyramid in which feature was detected
 * @param r feature row
 * @param c feature col
 * @param curv_thr high threshold on ratio of principal curvatures
 * @return false if the feature at (r,c) in dog_img is sufficiently corner-like or true otherwise.
 */
static bool is_too_edge_like( const cv::Mat& dog_img, unsigned short r, unsigned short c, unsigned short curv_thr );

/**
 * Calculates characteristic scale for each feature in an array.
 *
 * @param features array of features
 * @param sigma amount of Gaussian smoothing per octave of scale space
 * @param intvls intervals per octave of scale space
 */
static void calc_feature_scales( std::vector<SiftFeature*>& features, double sigma, unsigned short intvls );

/**
 * Halves feature coordinates and scale in case the input image was doubled prior to scale space construction.
 *
 * @param features array of features
 */
static void adjust_for_img_dbl( std::vector<SiftFeature*>& features );

/**
 * Computes a canonical orientation for each image feature in an array.  Based on Section 5 of Lowe's paper. This
 * function adds features to the array when there is more than one dominant orientation at a given feature location.
 *
 * @param features an array of image features
 * @param gauss_pyr Gaussian scale space pyramid
 */
static void calc_feature_oris( std::vector<SiftFeature*>& features, cv::Mat** const& gauss_pyr );

/**
 * Computes a gradient orientation histogram at a specified pixel.
 *
 * @param img image
 * @param r pixel row
 * @param c pixel col
 * @param n number of histogram bins
 * @param rad radius of region over which histogram is computed
 * @param sigma std for Gaussian weighting of histogram entries
 * @return an n-element array containing an orientation histogram representing orientations between 0 and 2 PI.
 */
static double* ori_hist( const cv::Mat& img, unsigned short r, unsigned short c, unsigned short n, int rad,
                         double sigma );

/**
 * Calculates the gradient magnitude and orientation at a given pixel.
 *
 * @param img image
 * @param r pixel row
 * @param c pixel col
 * @param mag output as gradient magnitude at pixel (r,c)
 * @param ori output as gradient orientation at pixel (r,c)
 * @return true if the specified pixel is a valid one and sets mag and ori accordingly; otherwise returns false
 */
static bool calc_grad_mag_ori( const cv::Mat& img, unsigned short r, unsigned short c, double* mag, double* ori );

/**
 * Gaussian smooths an orientation histogram.
 *
 * @param hist an orientation histogram
 * @param n number of bins
 */
static void smooth_ori_hist( double* hist, unsigned short n );

/**
 * Finds the magnitude of the dominant orientation in a histogram
 *
 * @param hist an orientation histogram
 * @param n number of bins
 * @return the value of the largest bin in hist
 */
static double dominant_ori( double* hist, unsigned short n );

/**
 * Interpolates a histogram peak from left, center, and right values
 */
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )

/**
 * Adds features to an array for every orientation in a histogram greater than a specified threshold.
 *
 * @param features new features are added to the end of this array
 * @param hist orientation histogram
 * @param n number of bins in hist
 * @param mag_thr new features are added for entries in hist greater than this
 * @param feat new features are clones of this with different orientations
 */
static void add_good_ori_features( std::vector<SiftFeature*>& features, double* hist, unsigned short n,
                                   double mag_thr, SiftFeature* feat );

/**
  Computes feature descriptors for features in an array.  Based on Section 6 of Lowe's paper.

  @param features array of features
  @param gauss_pyr Gaussian scale space pyramid
  @param d width of 2D array of orientation histograms
  @param n number of bins per orientation histogram
*/
static void compute_descriptors( std::vector<SiftFeature*>& features, cv::Mat** const& gauss_pyr, unsigned short d,
                                 unsigned short n );

/**
 * Computes the 2D array of orientation histograms that form the feature descriptor.
 * Based on Section 6.1 of Lowe's paper.
 *
 * @param img image used in descriptor computation
 * @param r row coord of center of orientation histogram array
 * @param c column coord of center of orientation histogram array
 * @param ori canonical orientation of feature whose descr is being computed
 * @param scl scale relative to img of feature whose descr is being computed
 * @param d width of 2d array of orientation histograms
 * @param n bins per orientation histogram
 * @return d x d array of n-bin orientation histograms.
 */
static double*** descr_hist( const cv::Mat& img, unsigned short r, unsigned short c, double ori, double scl,
                             unsigned short d, unsigned short n );

/**
 * Interpolates an entry into the array of orientation histograms that form the feature descriptor.
 *
 * @param hist 2D array of orientation histograms
 * @param rbin sub-bin row coordinate of entry
 * @param cbin sub-bin column coordinate of entry
 * @param obin sub-bin orientation coordinate of entry
 * @param mag size of entry
 * @param d width of 2D array of orientation histograms
 * @param n number of bins per orientation histogram
 */
static void interp_hist_entry( double*** hist, double rbin, double cbin,
                               double obin, double mag, unsigned short d, unsigned short n );

/**
 * Converts the 2D array of orientation histograms into a feature's descriptor vector.
 *
 * @param hist 2D array of orientation histograms
 * @param d width of hist
 * @param n bins per histogram
 * @param feat feature into which to store descriptor
 */
static void hist_to_descr( double*** hist, unsigned short d, unsigned short n, SiftFeature* feat );

/**
 * Normalizes a feature's descriptor vector to unitl length
 *
 * @param feat feature
 */
static void normalize_descr( SiftFeature* feat );

/**
 * De-allocates memory held by a descriptor histogram
 *
 * @param hist pointer to a 2D array of orientation histograms
 * @param d width of hist
 */
static void release_descr_hist( double***& hist, unsigned short d );

/**
 * De-allocates memory held by a scale space pyramid
 *
 * @param pyr scale space pyramid
 * @param octvs number of octaves of scale space
 * @param n number of images per octave
 */
static void release_pyr( cv::Mat**& pyr, unsigned short octvs, unsigned short n );


/************************************ IMPLEMENTATION ****************************************/


std::vector<SiftFeature*>  sift_features( cv::Mat const& img)
{
    return _sift_features( img, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR, SIFT_CURV_THR,
                           SIFT_IMG_DBL, SIFT_DESCR_WIDTH, SIFT_DESCR_HIST_BINS );
}


std::vector<SiftFeature*>  _sift_features( cv::Mat const& img, unsigned short intvls,
                                           double sigma, double contr_thr, unsigned short curv_thr,
                                           bool img_dbl, unsigned short descr_width, unsigned short descr_hist_bins )
{
    unsigned short octvs = 0;

    if ( img.empty() )
    {
        std::cerr << "Input image is empty." << std::endl;
        return std::vector<SiftFeature*>();
    }

    // build scale space pyramid; smallest dimension of top level is ~4 pixels
    cv::Mat init_img = create_init_img( img, img_dbl, sigma );

    octvs = log( MIN( init_img.cols, init_img.rows) ) / log(2) - 2;
    cv::Mat** gauss_pyr = build_gauss_pyr( init_img, octvs, intvls, sigma );
    cv::Mat** dog_pyr = build_dog_pyr( gauss_pyr, octvs, intvls );

    std::vector<SiftFeature*> features = scale_space_extrema( dog_pyr, octvs, intvls, contr_thr, curv_thr );
    calc_feature_scales( features, sigma, intvls );

    if ( img_dbl )
    {
        adjust_for_img_dbl( features );
    }
    calc_feature_oris( features, gauss_pyr );
    compute_descriptors( features, gauss_pyr, descr_width, descr_hist_bins );

    release_pyr( gauss_pyr, octvs, intvls + 3 );
    release_pyr( dog_pyr, octvs, intvls + 2 );
    return features;
}


static cv::Mat  create_init_img( cv::Mat const& img, bool img_dbl, double sigma )
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


static cv::Mat convert_to_gray32( cv::Mat const& img )
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


static cv::Mat** build_gauss_pyr(  cv::Mat const& base, unsigned short octvs, unsigned short intvls, double sigma )
{
    std::vector<double> sig;
    sig.resize(intvls + 3, 0);

    cv::Mat** gauss_pyr = new cv::Mat*[octvs];
    for (unsigned short i = 0; i < octvs; i++ )
    {
        gauss_pyr[i] = new cv::Mat[intvls + 3];
    }
    /*
     * Precompute Gaussian sigmas using the following formula:
     *
     * \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
     *
     * sig[i] is the incremental sigma value needed to compute the actual sigma of level i.
     * Keeping track of incremental sigmas vs. total sigmas keeps the gaussian kernel small.
     */
    double k = pow( 2.0, 1.0 / intvls );
    sig[0] = sigma;
    sig[1] = sigma * sqrt( k * k - 1 );
    for (unsigned short i = 2; i < intvls + 3; i++)
    {
        sig[i] = sig[i - 1] * k;
    }

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
            cv::GaussianBlur(gauss_pyr[o][i - 1], gauss_pyr[o][i], cv::Size(), sig[i], sig[i], cv::BORDER_REPLICATE );
        }
    }
    return gauss_pyr;
}


static cv::Mat** build_dog_pyr( cv::Mat** const& gauss_pyr, unsigned short octvs, unsigned short intvls )
{
    cv::Mat** dog_pyr = new cv::Mat*[octvs];

    for (int i = 0; i < octvs; i++ )
    {
        dog_pyr[i] = new cv::Mat[intvls + 2];
    }

    for (unsigned short o = 0; o < octvs; o++ )
    {
        for (unsigned short i = 0; i < intvls + 2; i++ )
        {
            dog_pyr[o][i] = cv::Mat(gauss_pyr[o][i].rows,gauss_pyr[o][i].cols, CV_32FC1);
            cv::subtract(gauss_pyr[o][i + 1], gauss_pyr[o][i], dog_pyr[o][i] );
        }
    }
    return dog_pyr;
}


static std::vector<SiftFeature*> scale_space_extrema( cv::Mat** const& dog_pyr, int octvs, int intvls,
                                                      double contr_thr, int curv_thr)
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
                        if ( is_extremum( dog_pyr, o, i, r, c ) )
                        {
                            SiftFeature* feat = interp_extremum(dog_pyr, o, i, r, c, intvls, contr_thr);
                            if ( feat )
                            {
                                if ( ! is_too_edge_like( dog_pyr[feat->ddata_octv][feat->ddata_intvl],
                                                         feat->ddata_r, feat->ddata_c, curv_thr ) )
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


static bool is_extremum( cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl,
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


static SiftFeature* interp_extremum( cv::Mat** const& dog_pyr, unsigned short octv,  unsigned short intvl,
                                     unsigned short r, unsigned short c, unsigned short intvls, double contr_thr )
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


static void interp_step( cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl, unsigned short r,
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


static cv::Matx31d deriv_3D( cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl,
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


static cv::Matx33d hessian_3D( cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl,
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


static double interp_contr( cv::Mat** const& dog_pyr, unsigned short octv, unsigned short intvl,
                            unsigned short r, unsigned short c, double xi, double xr, double xc )
{
    cv::Matx31d X(xc,xr,xi);
    cv::Matx<double,1,1> T;
    cv::Matx31d dD = deriv_3D( dog_pyr, octv, intvl, r, c );

    cv::gemm( dD, X, 1, cv::Matx<double,0,0>(), 0, T,  cv::GEMM_1_T );

    return dog_pyr[octv][intvl].at<float>( r, c ) + T(0) * 0.5;
}


static bool is_too_edge_like( const cv::Mat& dog_img, unsigned short r, unsigned short c, unsigned short curv_thr )
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


static void calc_feature_scales( std::vector<SiftFeature*>& features, double sigma, unsigned short intvls )
{
    unsigned short n = features.size();
    for (unsigned short i = 0; i < n; i++ )
    {
        SiftFeature* feat = features[i];
        double intvl = feat->ddata_intvl + feat->ddata_subintvl;
        feat->scl = sigma * pow( 2.0, feat->ddata_octv + intvl / intvls );
        feat->ddata_scl_octv = sigma * pow( 2.0, intvl / intvls );
    }
}


static void adjust_for_img_dbl( std::vector<SiftFeature*>& features )
{
    unsigned short n = features.size();
    for (unsigned short  i = 0; i < n; i++ )
    {
        SiftFeature* feat = features[i];
        feat->x /= 2.0;
        feat->y /= 2.0;
        feat->scl /= 2.0;
        feat->img_pt.x /= 2.0;
        feat->img_pt.y /= 2.0;
    }
}


static void calc_feature_oris( std::vector<SiftFeature*>& features, cv::Mat** const& gauss_pyr )
{
    unsigned short n = features.size();

    for (unsigned short i = 0; i < n; i++ )
    {
        SiftFeature* feat =  features[i];
        double* hist = ori_hist( gauss_pyr[feat->ddata_octv][feat->ddata_intvl],
                feat->ddata_r, feat->ddata_c, SIFT_ORI_HIST_BINS,
                cvRound( SIFT_ORI_RADIUS * feat->ddata_scl_octv ),
                SIFT_ORI_SIG_FCTR * feat->ddata_scl_octv );
        for (unsigned short j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++ )
        {
            smooth_ori_hist( hist, SIFT_ORI_HIST_BINS );
        }
        double omax = dominant_ori( hist, SIFT_ORI_HIST_BINS );
        add_good_ori_features( features, hist, SIFT_ORI_HIST_BINS,
                               omax * SIFT_ORI_PEAK_RATIO, feat );
        delete feat;
        delete[] hist;
    }
    features.erase(features.begin(), features.begin() + n);
}

#include <array>

static double* ori_hist( const cv::Mat& img, unsigned short r, unsigned short c, unsigned short n, int rad,
                         double sigma )
{
    double mag, ori, PI2 = CV_PI * 2.0;



    double* hist = new double[n];
    memset(hist,0,n*sizeof(double));

    double exp_denom = 2.0 * sigma * sigma;
    for (int  i = -rad; i <= rad; i++ )
    {
        for (int  j = -rad; j <= rad; j++ )
        {
            if ( calc_grad_mag_ori( img, r + i, c + j, &mag, &ori ) )
            {
                double w = exp( -( i * i + j * j ) / exp_denom );
                int bin = cvRound( n * ( ori + CV_PI ) / PI2 );
                bin = ( bin < n ) ? bin : 0;
                hist[bin] += w * mag;
            }
        }
    }
    return hist;
}


static bool calc_grad_mag_ori( const cv::Mat& img, unsigned short r, unsigned short c, double* mag, double* ori )
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


static void smooth_ori_hist( double* hist, unsigned short n )
{
    double h0 = hist[0];

    double prev = hist[n - 1];
    for (unsigned short i = 0; i < n; i++ )
    {
        double tmp = hist[i];
        hist[i] = 0.25 * prev + 0.5 * hist[i] +
                  0.25 * ( ( i + 1 == n ) ? h0 : hist[i + 1] );
        prev = tmp;
    }
}


static double dominant_ori( double* hist, unsigned short n )
{
    double omax = hist[0];
    //int maxbin = 0;
    for (unsigned short i = 1; i < n; i++ )
    {
        if ( hist[i] > omax )
        {
            omax = hist[i];
            //maxbin = i;
        }
    }
    return omax;
}


static void add_good_ori_features( std::vector<SiftFeature*>& features, double* hist, unsigned short n,
                                   double mag_thr, SiftFeature* feat )
{
    double PI2 = CV_PI * 2.0;

    for (unsigned short i = 0; i < n; i++ )
    {
        int l = ( i == 0 ) ? n - 1 : i - 1;
        int r = ( i + 1 ) % n;

        if ( hist[i] > hist[l]  &&  hist[i] > hist[r]  &&  hist[i] >= mag_thr )
        {
            double bin = i + interp_hist_peak( hist[l], hist[i], hist[r] );
            bin = ( bin < 0 ) ? n + bin : ( bin >= n ) ? bin - n : bin;
            SiftFeature* new_feat = new SiftFeature( *feat );
            new_feat->ori = ( ( PI2 * bin ) / n ) - CV_PI;
            features.push_back(new_feat);
        }
    }
}


static void compute_descriptors( std::vector<SiftFeature*>& features, cv::Mat** const& gauss_pyr, unsigned short d,
                                 unsigned short n )
{
    unsigned short k = features.size();

    for (unsigned short i = 0; i < k; i++ )
    {
        SiftFeature* feat = features[i];
        double*** hist = descr_hist( cv::Mat(gauss_pyr[feat->ddata_octv][feat->ddata_intvl]), feat->ddata_r,
                feat->ddata_c, feat->ori, feat->ddata_scl_octv, d, n );
        hist_to_descr( hist, d, n, feat );
        release_descr_hist( hist, d );
    }
}


static double*** descr_hist( const cv::Mat& img, unsigned short r, unsigned short c, double ori, double scl,
                             unsigned short d, unsigned short n )
{
    double grad_mag, grad_ori, PI2 = 2.0 * CV_PI;

    double*** hist = new double**[d];
    for (unsigned short i = 0; i < d; i++ )
    {
        hist[i] = new double*[d];
        for (int j = 0; j < d; j++ )
        {
            hist[i][j] = new double[n];
            memset(hist[i][j],0,n*sizeof(double));
        }
    }

    double cos_t = cos( ori );
    double sin_t = sin( ori );
    double bins_per_rad = n / PI2;
    double exp_denom = d * d * 0.5;
    double hist_width = SIFT_DESCR_SCL_FCTR * scl;
    int radius = hist_width * sqrt(2) * ( d + 1.0 ) * 0.5 + 0.5;
    for (int  i = -radius; i <= radius; i++ )
    {
        for (int j = -radius; j <= radius; j++ )
        {
            /* Calculate sample's histogram array coords rotated relative to ori.
               Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
               r_rot = 1.5) have full weight placed in row 1 after interpolation. */

            double c_rot = ( j * cos_t - i * sin_t ) / hist_width;
            double r_rot = ( j * sin_t + i * cos_t ) / hist_width;
            double rbin = r_rot + d / 2 - 0.5;
            double cbin = c_rot + d / 2 - 0.5;

            if ( rbin > -1.0  &&  rbin < d  &&  cbin > -1.0  &&  cbin < d )
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
                    interp_hist_entry( hist, rbin, cbin, obin, grad_mag * w, d, n );
                }
            }
        }
    }

    return hist;
}


static void interp_hist_entry( double*** hist, double rbin, double cbin,
                               double obin, double mag, unsigned short d, unsigned short n )
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
        if ( rb >= 0  &&  rb < d )
        {
            double v_r = mag * ( ( r == 0 ) ? 1.0 - d_r : d_r );
            double** row = hist[rb];
            for (int c = 0; c <= 1; c++ )
            {
                int cb = c0 + c;
                if ( cb >= 0  &&  cb < d )
                {
                    double v_c = v_r * ( ( c == 0 ) ? 1.0 - d_c : d_c );
                    double* h = row[cb];
                    for (int o = 0; o <= 1; o++ )
                    {
                        int ob = ( o0 + o ) % n;
                        double v_o = v_c * ( ( o == 0 ) ? 1.0 - d_o : d_o );
                        h[ob] += v_o;
                    }
                }
            }
        }
    }
}


static void hist_to_descr( double*** hist, unsigned short d, unsigned short n, SiftFeature* feat )
{
    unsigned short k = 0;

    for (unsigned short  r = 0; r < d; r++ )
    {
        for (unsigned short  c = 0; c < d; c++ )
        {
            for (unsigned short  o = 0; o < n; o++ )
            {
                feat->descr[k++] = hist[r][c][o];
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


static void normalize_descr( SiftFeature* feat )
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


static void release_descr_hist( double***& hist, unsigned short d )
{
    for (unsigned short  i = 0; i < d; i++)
    {
        for (unsigned short  j = 0; j < d; j++ )
        {
            delete[] hist[i][j];
        }
        delete[] hist[i];
    }
    delete[] hist;
    hist = nullptr;
}


static void release_pyr( cv::Mat**& pyr, unsigned short octvs, unsigned short n )
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

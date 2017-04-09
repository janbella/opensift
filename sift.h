#ifndef SIFT_H
#define SIFT_H

/**
 * @file sift.h
 * @author Ján Bella <xbella1@fi.muni.cz>
 *
 * The file provides the definition of SIFT class, which is an object for computing SIFT keypoints and descriptors
 *
 * The implementation is based on Rob Hess's opensift, see the original copyright below.
 */

/**
   Functions for detecting SIFT image features.

   For more information, refer to:

   Lowe, D.  Distinctive image features from scale-invariant keypoints.
   <EM>International Journal of Computer Vision, 60</EM>, 2 (2004),  pp.91--110.

   Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

   Note: The SIFT algorithm is patented in the United States and cannot be
   used in commercial products without a license from the University of
   British Columbia.  For more information, refer to the file LICENSE.ubc
   that accompanied this distribution.

   @version 1.1.2-20100521
*/

#include <opencv2/opencv.hpp>
#include "siftfeature.h"

namespace Stitching
{

/******************************* Defs and macros *****************************/
// default number of sampled intervals per octave
#define SIFT_INTVLS 3

// default sigma for initial gaussian smoothing
#define SIFT_SIGMA 1.6

// default threshold on keypoint contrast |D(x)|
#define SIFT_CONTR_THR 0.04

// default threshold on keypoint ratio of principle curvatures
#define SIFT_CURV_THR 10

// double image size before pyramid construction?
#define SIFT_IMG_DBL true

// default width of descriptor histogram array
#define SIFT_DESCR_WIDTH 4

// default number of bins per histogram in descriptor array
#define SIFT_DESCR_HIST_BINS 8

// assumed gaussian blur for input image
#define SIFT_INIT_SIGMA 0.5

// width of border in which to ignore keypoints
#define SIFT_IMG_BORDER 2

// maximum steps of keypoint interpolation before failure
#define SIFT_MAX_INTERP_STEPS 5

// default number of bins in histogram for orientation assignment
#define SIFT_ORI_HIST_BINS 36

// determines gaussian sigma for orientation assignment
#define SIFT_ORI_SIG_FCTR 1.5

// determines the radius of the region used in orientation assignment
#define SIFT_ORI_RADIUS 3.0 * SIFT_ORI_SIG_FCTR

// number of passes of orientation histogram smoothing
#define SIFT_ORI_SMOOTH_PASSES 2

// orientation magnitude relative to max that results in new feature
#define SIFT_ORI_PEAK_RATIO 0.8

// determines the size of a single descriptor orientation histogram
#define SIFT_DESCR_SCL_FCTR 3.0

// threshold on magnitude of elements of descriptor vector
#define SIFT_DESCR_MAG_THR 0.2

// factor used to convert floating-point descriptor to unsigned char
#define SIFT_INT_DESCR_FCTR 512.0

class SIFT
{
private:
    unsigned short octvs;
    unsigned short intvls;
    double sigma;
    double contr_thr;
    unsigned short curv_thr;
    bool img_dbl;
    unsigned short descr_width;
    unsigned short descr_hist_bins;

    cv::Mat** gauss_pyr;
    cv::Mat** dog_pyr;

    double* gaussPyrIncrSig;
    double* orientationHistogram;
    double*** descriptorHistogram;

public:

    /**
     * Default constructor, sets the attributes based on the defines on the top of this file
     */
    SIFT();

    /**
     * Parametric constructor
     * @param intvls
     * @param sigma
     * @param contr_thr
     * @param curv_thr
     * @param img_dbl
     * @param descr_width
     * @param descr_hist_bins
     */
    SIFT(unsigned short intvls, double sigma, double contr_thr, unsigned short curv_thr,
         bool img_dbl, unsigned short descr_width, unsigned short descr_hist_bins);

    /**
     * Copy constructor.
     * @param other
     */
    SIFT(const SIFT& other);

    /**
     * Main method, returns keypoints found in img.
     * @param img
     * @return
     */
    std::vector<SiftFeature*> extractFeatures( cv::Mat& img);

    /**
     * Desctructor, releases used memory
     */
    virtual ~SIFT();

    /**
     * Returns recommended number of octaves for image of size rows x cols
     * @param rows
     * @param cols
     * @return
     */
    static unsigned short recommendedOctaves(unsigned short rows, unsigned short cols);

    /**
     * Returns recommended number of octaves for the given image
     * @param image
     * @return
     */
    static unsigned short recommendedOctaves(const cv::Mat& image);

private:
    /**
     * Allocate memory for the scale-space pyramid \a pyr
     * @param pyr
     * @param n
     */
    void alloc_pyr( cv::Mat**& pyr, unsigned short n );

    /**
     * Converts an image to 8-bit grayscale and Gaussian-smooths it. The image is
     * optionally doubled in size prior to smoothing.
     *
     * @param img input image
     */
    cv::Mat  create_init_img( cv::Mat const& img );

    /**
     * Converts an image to 32-bit grayscale
     *
     * @param img a 3-channel 8-bit color (BGR) or 8-bit gray image
     * @return 32-bit grayscale image
     */
    cv::Mat convert_to_gray32( cv::Mat const& img );

    /**
     * Builds Gaussian scale space pyramid from an image
     *
     * @param base base image of the pyramid
     */
    void build_gauss_pyr(  cv::Mat const& base );

    /**
     * Builds a difference of Gaussians scale space pyramid by subtracting adjacent intervals of a Gaussian pyramid
     */
    void build_dog_pyr( );

    /**
     * Detects features at extrema in DoG scale space.  Bad features are discarded based on contrast
     * and ratio of principal curvatures.
     *
     * @return Returns an array of detected features whose scales, orientations, and descriptors are yet to be determined.
     */
    std::vector<SiftFeature*> scale_space_extrema( );

    /**
     * Determines whether a pixel is a scale-space extremum by comparing it to it's 3x3x3 pixel neighborhood.
     *
     * @param pyr scale space pyramid
     * @param octv pixel's scale space octave
     * @param intvl pixel's within-octave interval
     * @param r pixel's image row
     * @param c pixel's image col
     * @return true if the specified pixel is an extremum (max or min) among it's 3x3x3 pixel neighborhood.
     */
    bool is_extremum(cv::Mat** const& pyr, unsigned short octv, unsigned short intvl, unsigned short r, unsigned short c );

    /**
     * Interpolates a scale-space extremum's location and scale to subpixel accuracy to form an image feature.
     * Rejects features with low contrast. Based on Section 4 of Lowe's paper.
     *
     * @param pyr DoG scale space pyramid
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
    SiftFeature* interp_extremum(cv::Mat** const& pyr,  unsigned short octv,  unsigned short intvl, unsigned short r, unsigned short c );

    /**
     * Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's paper.
     *
     * @param pyr scale space pyramid
     * @param octv octave of scale space
     * @param intvl interval being interpolated
     * @param r row being interpolated
     * @param c column being interpolated
     * @param xi output as interpolated subpixel increment to interval
     * @param xr output as interpolated subpixel increment to row
     * @param xc output as interpolated subpixel increment to col
     */

    void interp_step(cv::Mat** const& pyr,  unsigned short octv, unsigned short intvl, unsigned short r,
                      unsigned short c, double* xi, double* xr, double* xc );

    /**
     * Computes the partial derivatives in x, y, and scale of a pixel in the DoG scale space pyramid.
     *
     * @param pyr DoG scale space pyramid
     * @param octv pixel's octave in dog_pyr
     * @param intvl pixel's interval in octv
     * @param r pixel's image row
     * @param c pixel's image col
     * @return vector of partial derivatives for pixel I { dI/dx, dI/dy, dI/ds }^T as a cv::Matx31d
     */
    cv::Matx31d deriv_3D(cv::Mat** const& pyr,  unsigned short octv, unsigned short intvl,
                          unsigned short r, unsigned short c );

    /**
     * Computes the 3D Hessian matrix for a pixel in the DoG scale space pyramid.
     *
     * @param pyr DoG scale space pyramid
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
    cv::Matx33d hessian_3D(cv::Mat** const& pyr, unsigned short octv, unsigned short intvl,
                            unsigned short r, unsigned short c );

    /**
     * Calculates interpolated pixel contrast.  Based on Eqn. (3) in Lowe's paper.
     *
     * @param pyr scale space pyramid
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
    double interp_contr(cv::Mat** const& pyr,  unsigned short octv, unsigned short intvl,
                         unsigned short r, unsigned short c, double xi, double xr, double xc );

    /**
     * Determines whether a feature is too edge like to be stable by computing the ratio of principal
     *    curvatures at that feature.  Based on Section 4.1 of Lowe's paper.
     *
     * @param dog_img image from the DoG pyramid in which feature was detected
     * @param r feature row
     * @param c feature col
     * @return false if the feature at (r,c) in dog_img is sufficiently corner-like or true otherwise.
     */
    bool is_too_edge_like(cv::Mat const& dog_img,  unsigned short r, unsigned short c);

    /**
     * Calculates characteristic scale for each feature in an array.
     *
     * @param features array of features
     */
    void calc_feature_scales( std::vector<SiftFeature*>& features);

    /**
     * Halves feature coordinates and scale in case the input image was doubled prior to scale space construction.
     *
     * @param features array of features
     */
    void adjust_for_img_dbl( std::vector<SiftFeature*>& features );

    /**
     * Computes a canonical orientation for each image feature in an array.  Based on Section 5 of Lowe's paper. This
     * function adds features to the array when there is more than one dominant orientation at a given feature location.
     *
     * @param features an array of image features
     */
    void calc_feature_oris( std::vector<SiftFeature*>& features);

    /**
     * Computes a gradient orientation histogram at a specified pixel.
     *
     * @param img image
     * @param r pixel row
     * @param c pixel col
     * @param rad radius of region over which histogram is computed
     * @param sigma std for Gaussian weighting of histogram entries
     */
    void compute_ori_hist( const cv::Mat& img, unsigned short r, unsigned short c, int rad, double sigma );

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
    bool calc_grad_mag_ori( const cv::Mat& img, unsigned short r, unsigned short c, double* mag, double* ori );

    /**
     * Gaussian smooths an orientation histogram.
     */
    void smooth_ori_hist();

    /**
     * Finds the magnitude of the dominant orientation in a histogram
     *
     * @return the value of the largest bin in hist
     */
    double dominant_ori();

    /**
     * Interpolates a histogram peak from left, center, and right values
     */
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )

    /**
     * Adds features to an array for every orientation in a histogram greater than a specified threshold.
     *
     * @param features new features are added to the end of this array
     * @param mag_thr new features are added for entries in hist greater than this
     * @param feat new features are clones of this with different orientations
     */
    void add_good_ori_features( std::vector<SiftFeature*>& features, double mag_thr, SiftFeature* feat );

   /**
    * Computes feature descriptors for features in an array.  Based on Section 6 of Lowe's paper.
    *
    * @param features array of features
    * @param gauss_pyr Gaussian scale space pyramid
    */
    void compute_descriptors( std::vector<SiftFeature*>& features);

    /**
     * Computes the 2D array of orientation histograms that form the feature descriptor.
     * Based on Section 6.1 of Lowe's paper.
     *
     * @param img image used in descriptor computation
     * @param r row coord of center of orientation histogram array
     * @param c column coord of center of orientation histogram array
     * @param ori canonical orientation of feature whose descr is being computed
     * @param scl scale relative to img of feature whose descr is being computed
     * @return d x d array of n-bin orientation histograms.
     */
    void descr_hist( const cv::Mat& img, unsigned short r, unsigned short c, double ori, double scl);

    /**
     * Interpolates an entry into the array of orientation histograms that form the feature descriptor.
     *
     * @param rbin sub-bin row coordinate of entry
     * @param cbin sub-bin column coordinate of entry
     * @param obin sub-bin orientation coordinate of entry
     * @param mag size of entry
     */
    void interp_desc_hist_entry( double rbin, double cbin, double obin, double mag);

    /**
     * Converts the 2D array of orientation histograms into a feature's descriptor vector.
     *
     * @param feat feature into which to store descriptor
     */
    void hist_to_descr(  SiftFeature* feat );

    /**
     * Normalizes a feature's descriptor vector to unitl length
     *
     * @param feat feature
     */
    void normalize_descr( SiftFeature* feat );

    /**
     * De-allocates memory held by a descriptor histogram
     */
    void release_descr_hist( );

    /**
     * De-allocates memory held by a scale space pyramid
     *
     * @param pyr scale space pyramid
     * @param octvs number of octaves of scale space
     * @param n number of images per octave
     */
    void release_pyr( cv::Mat**& pyr, unsigned short n );

    /**
     * Precompute Gaussian sigmas using the following formula:
     *
     * \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
     *
     * sig[i] is the incremental sigma value needed to compute the actual sigma of level i.
     * Keeping track of incremental sigmas vs. total sigmas keeps the gaussian kernel small.
     */
    void precompute_incremental_sigmas();

    void alloc_descr_hist();
};
}
#endif // SIFT_H

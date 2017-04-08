#ifndef OPENSIFT_C_H
#define OPENSIFT_C_H

/**
 * @file sift_c.h
 * @author Rob Hess, edited by Ján Bella.
 *
 * See original copyright below.
 **/

/**
   Functions for detecting SIFT image features.

   For more information, refer to:

   Lowe, D.  Distinctive image features from scale-invariant keypoints.
   <EM>International Journal of Computer Vision, 60</EM>, 2 (2004),
   pp.91--110.

   Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

   Note: The SIFT algorithm is patented in the United States and cannot be
   used in commercial products without a license from the University of
   British Columbia.  For more information, refer to the file LICENSE.ubc
   that accompanied this distribution.

   @version 1.1.2-20100521
*/

#include <opencv2/opencv.hpp>
#include "siftfeature.h"

using Stitching::SiftFeature;

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

// absolute value
#ifndef ABS
#define ABS(x) ( ( (x) < 0 )? -(x) : (x) )
#endif


/*************************** Function Prototypes *****************************/

/**
 *  Finds SIFT features in an image using default parameter values.
 *
 *  @param img the image in which to detect features
 *  @param feat a pointer to an array in which to store detected features
 *
 *  @return Returns the vector of found keypoints. Empty vector on failure.
 *  @see _sift_features()
 */
extern std::vector<SiftFeature*>  sift_features( cv::Mat const& img);



/**
 *  Finds SIFT features in an image using user-specified parameter values.  All
 *  detected features are stored in the array pointed to by \a feat.
 *
 *  @param img the image in which to detect features
 *  @param intvls the number of intervals sampled per octave of scale space
 *  @param sigma the amount of Gaussian smoothing applied to each image level
 *    before building the scale space representation for an octave
 *  @param cont_thr a threshold on the value of the scale space function
 *    \f$\left|D(\hat{x})\right|\f$, where \f$\hat{x}\f$ is a vector specifying
 *    feature location and scale, used to reject unstable features;  assumes
 *    pixel values in the range [0, 1]
 *  @param curv_thr threshold on a feature's ratio of principle curvatures
 *    used to reject features that are too edge-like
 *  @param img_dbl should be 1 if image doubling prior to scale space construction is desired or 0 if not
 *  @param descr_width the width, \f$n\f$, of the \f$n \times n\f$ array of
 *    orientation histograms used to compute a feature's descriptor
 *  @param descr_hist_bins the number of orientations in each of the
 *    histograms in the array used to compute a feature's descriptor
 *
 *  @return Returns the vector of found keypoints. Empty vector on failure.
 */
extern std::vector<SiftFeature*>  _sift_features( cv::Mat const& img, unsigned short intvls,
                                                  double sigma, double contr_thr, unsigned short curv_thr,
                                                  bool img_dbl, unsigned short descr_width, unsigned short descr_hist_bins );

#endif // OPENSIFT_C_H

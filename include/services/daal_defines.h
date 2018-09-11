/* file: daal_defines.h */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Common definitions.
//--
*/

#ifndef __DAAL_DEFINES_H__
#define __DAAL_DEFINES_H__

/** \file daal_defines.h */

#include <cstddef> // for size_t

#if defined(_WIN32) || defined(_WIN64)
  #ifdef __DAAL_IMPLEMENTATION
    #define DAAL_EXPORT __declspec( dllexport )
  #else
    #define DAAL_EXPORT
  #endif
#else
  #define DAAL_EXPORT
#endif

#if (defined(__INTEL_CXX11_MODE__) || __cplusplus > 199711L)
  #define DAAL_C11_OVERRIDE override
#else
  #define DAAL_C11_OVERRIDE
#endif

/* Intel(R) DAAL 64-bit integer types */
#if (!defined(__INTEL_COMPILER)) & defined(_MSC_VER)
  #define DAAL_INT64 __int64
  #define DAAL_UINT64 unsigned __int64
#else
  #define DAAL_INT64 long long int
  #define DAAL_UINT64 unsigned long long int
#endif

#if !defined(DAAL_INT)
  #if defined(_WIN64) || defined(__x86_64__)
    #define DAAL_INT __int64
  #else
    #define DAAL_INT __int32
  #endif
#endif

#if defined DAAL_HIDE_DEPRECATED
  #define DAAL_DEPRECATED
#else
  #ifdef __GNUC__
    #define DAAL_DEPRECATED __attribute__ ((deprecated))
  #elif defined(_MSC_VER)
    #define DAAL_DEPRECATED __declspec(deprecated)
  #else
    #define DAAL_DEPRECATED
  #endif
#endif

#if defined DAAL_HIDE_DEPRECATED
  #define DAAL_DEPRECATED_VIRTUAL
#else
  #ifdef __INTEL_COMPILER
    #define DAAL_DEPRECATED_VIRTUAL DAAL_DEPRECATED
  #else
    #define DAAL_DEPRECATED_VIRTUAL
  #endif
#endif

#if defined(_MSC_VER)
    #define DAAL_FORCEINLINE __forceinline
#else
    #define DAAL_FORCEINLINE inline __attribute__((always_inline))
#endif

#if (defined __APPLE__)
    #define DAAL_CPU_TOPO_DISABLED
    #define DAAL_THREAD_PINNING_DISABLED
#endif

/**
 *  Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) namespace
 */
namespace daal
{
/**
* <a name="DAAL-ENUM-COMPUTEMODE"></a>
* Computation modes of Intel(R) DAAL algorithms
*/
enum ComputeMode
{
    batch                  = 1,    /*!< Batch processing computation mode */
    distributed            = 2,    /*!< Processing of data sets distributed across several devices */
    online                 = 4     /*!< Online mode - processing of data sets in blocks */
};

/**
 * <a name="DAAL-ENUM-__COMPUTESTEP"></a>
 * Describes computation steps in the distributed processing mode
 */
enum ComputeStep
{
    step1Local     = 0,        /*!< First step of the distributed processing mode */
    step2Master    = 1,        /*!< Second step of the distributed processing mode */
    step3Local     = 2,        /*!< Third step of the distributed processing mode */
    step4Local     = 3,        /*!< Fourth step of the distributed processing mode */

    step2Local     = 5,        /*!< Second step of the distributed processing mode performed by local node*/
    step3Master    = 6,        /*!< Third step of the distributed processing mode performed by master node*/
    step5Master    = 7         /*!< Fifth step of the distributed processing mode performed by master node*/
};

/**
 * <a name="DAAL-ENUM-MEMTYPE"></a>
 * Describes types of memory
 */
enum MemType
{
    dram   = 0,    /*!< DRAM */
    mcdram = 1     /*!< Multi-Channel DRAM */
};

typedef unsigned char byte;

/**
 * <a name="DAAL-STRUCT-ISSAMETYPE"></a>
 * Indicates if the types are not equal
 */
template<class U, class V> struct IsSameType
{ static const bool value     = false; };

/**
 * <a name="DAAL-STRUCT-ISSAMETYPE"></a>
 * Indicates if the types are equal
 */
template<class U>          struct IsSameType<U, U>
{ static const bool value     = true; };

const size_t DAAL_MALLOC_DEFAULT_ALIGNMENT = 64;

const int SERIALIZATION_HOMOGEN_NT_ID                                                          = 1000;
const int SERIALIZATION_AOS_NT_ID                                                              = 3000;
const int SERIALIZATION_SOA_NT_ID                                                              = 3001;
const int SERIALIZATION_DATACOLLECTION_ID                                                      = 4000;
const int SERIALIZATION_KEYVALUEDATACOLLECTION_ID                                              = 4010;
const int SERIALIZATION_DATAFEATURE_NT_ID                                                      = 5000;
const int SERIALIZATION_DATADICTIONARY_NT_ID                                                   = 6000;
const int SERIALIZATION_DATADICTIONARY_DS_ID                                                   = 6010;
const int SERIALIZATION_MATRIX_NT_ID                                                           = 7000;
const int SERIALIZATION_CSR_NT_ID                                                              = 8000;
const int SERIALIZATION_PACKEDSYMMETRIC_NT_ID                                                  = 11000;
const int SERIALIZATION_PACKEDTRIANGULAR_NT_ID                                                 = 12000;
const int SERIALIZATION_MERGE_NT_ID                                                            = 13000;
const int SERIALIZATION_ROWMERGE_NT_ID                                                         = 14000;

const int SERIALIZATION_HOMOGEN_TENSOR_ID                                                      = 20000;
const int SERIALIZATION_TENSOR_OFFSET_LAYOUT_ID                                                = 22000;
const int SERIALIZATION_MKL_TENSOR_ID                                                          = 24000;

const int SERIALIZATION_OPTIONAL_RESULT_ID                                                     = 30000;
const int SERIALIZATION_MEMORY_BLOCK_ID                                                        = 40000;

const int SERIALIZATION_LINEAR_REGRESSION_MODELNORMEQ_ID                                       = 100100;
const int SERIALIZATION_LINEAR_REGRESSION_MODELQR_ID                                           = 100110;
const int SERIALIZATION_LINEAR_REGRESSION_PARTIAL_RESULT_ID                                    = 100120;
const int SERIALIZATION_LINEAR_REGRESSION_TRAINING_RESULT_ID                                   = 100130;
const int SERIALIZATION_LINEAR_REGRESSION_PREDICTION_RESULT_ID                                 = 100140;
const int SERIALIZATION_LINEAR_REGRESSION_SINGLE_BETA_RESULT_ID                                = 100150;
const int SERIALIZATION_LINEAR_REGRESSION_GROUP_OF_BETAS_RESULT_ID                             = 100160;

const int SERIALIZATION_PCA_RESULT_ID                                                          = 100200;
const int SERIALIZATION_PCA_PARTIAL_RESULT_CORRELATION_ID                                      = 100210;
const int SERIALIZATION_PCA_PARTIAL_RESULT_SVD_ID                                              = 100220;
const int SERIALIZATION_PCA_TRANSFORM_RESULT_ID                                                = 100230;
const int SERIALIZATION_PCA_QUALITY_METRIC_RESULT_ID                                           = 100240;

const int SERIALIZATION_STUMP_MODEL_ID                                                         = 100300;
const int SERIALIZATION_STUMP_TRAINING_RESULT_ID                                               = 100310;

const int SERIALIZATION_ADABOOST_MODEL_ID                                                      = 100400;
const int SERIALIZATION_ADABOOST_TRAINING_RESULT_ID                                            = 100410;

const int SERIALIZATION_BROWNBOOST_MODEL_ID                                                    = 100500;
const int SERIALIZATION_BROWNBOOST_TRAINING_RESULT_ID                                          = 100510;

const int SERIALIZATION_LOGITBOOST_MODEL_ID                                                    = 100600;
const int SERIALIZATION_LOGITBOOST_TRAINING_RESULT_ID                                          = 100610;

const int SERIALIZATION_NAIVE_BAYES_MODEL_ID                                                   = 100700;
const int SERIALIZATION_NAIVE_BAYES_PARTIALMODEL_ID                                            = 100710;
const int SERIALIZATION_NAIVE_BAYES_RESULT_ID                                                  = 100720;
const int SERIALIZATION_NAIVE_BAYES_PARTIAL_RESULT_ID                                          = 100730;

const int SERIALIZATION_SVM_MODEL_ID                                                           = 100800;
const int SERIALIZATION_SVM_TRAINING_RESULT_ID                                                 = 100810;

const int SERIALIZATION_MULTI_CLASS_CLASSIFIER_MODEL_ID                                        = 100900;
const int SERIALIZATION_MULTICLASS_CLASSIFIER_RESULT_ID                                        = 100910;

const int SERIALIZATION_COVARIANCE_PARTIAL_RESULT_ID                                           = 101000;
const int SERIALIZATION_COVARIANCE_RESULT_ID                                                   = 101010;

const int SERIALIZATION_KMEANS_PARTIAL_RESULT_ID                                               = 101100;
const int SERIALIZATION_KMEANS_RESULT_ID                                                       = 101110;
const int SERIALIZATION_KMEANS_INIT_PARTIAL_RESULT_ID                                          = 101200;
const int SERIALIZATION_KMEANS_INIT_STEP2LOCAL_PP_PARTIAL_RESULT_ID                            = 101210;
const int SERIALIZATION_KMEANS_INIT_STEP3MASTER_PP_PARTIAL_RESULT_ID                           = 101220;
const int SERIALIZATION_KMEANS_INIT_STEP4LOCAL_PP_PARTIAL_RESULT_ID                            = 101230;
const int SERIALIZATION_KMEANS_INIT_STEP5MASTER_PP_PARTIAL_RESULT_ID                           = 101240;

const int SERIALIZATION_KMEANS_INIT_RESULT_ID                                                  = 101300;

const int SERIALIZATION_CLASSIFIER_TRAINING_PARTIAL_RESULT_ID                                  = 101400;
const int SERIALIZATION_CLASSIFIER_BINARY_CONFUSION_MATRIX_RESULT_ID                           = 101410;
const int SERIALIZATION_CLASSIFIER_MULTICLASS_CONFUSION_MATRIX_RESULT_ID                       = 101420;
const int SERIALIZATION_CLASSIFIER_PREDICTION_RESULT_ID                                        = 101430;
const int SERIALIZATION_CLASSIFIER_TRAINING_RESULT_ID                                          = 101440;

const int SERIALIZATION_MOMENTS_PARTIAL_RESULT_ID                                              = 101500;
const int SERIALIZATION_MOMENTS_RESULT_ID                                                      = 101510;

const int SERIALIZATION_IMPLICIT_ALS_MODEL_ID                                                  = 101600;
const int SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID                                           = 101610;
const int SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_PARTIAL_RESULT_ID                      = 101620;
const int SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_RESULT_ID                              = 101630;
const int SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_RESULT_ID                                   = 101640;
const int SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_PARTIAL_RESULT_BASE_ID                      = 101645;
const int SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_PARTIAL_RESULT_ID                           = 101650;
const int SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID         = 101657;
const int SERIALIZATION_IMPLICIT_ALS_TRAINING_RESULT_ID                                        = 101660;
const int SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP1_ID                       = 101670;
const int SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP2_ID                       = 101675;
const int SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID                       = 101680;
const int SERIALIZATION_IMPLICIT_ALS_DISTRIBUTED_PARTIAL_RESULT_STEP4_ID                       = 101685;

const int SERIALIZATION_ASSOCIATION_RULES_RESULT_ID                                            = 101700;

const int SERIALIZATION_CHOLESKY_RESULT_ID                                                     = 101800;

const int SERIALIZATION_CORRELATION_DISTANCE_RESULT_ID                                         = 101900;
const int SERIALIZATION_COSINE_DISTANCE_RESULT_ID                                              = 101910;

const int SERIALIZATION_EM_GMM_INIT_RESULT_ID                                                  = 102000;
const int SERIALIZATION_EM_GMM_RESULT_ID                                                       = 102010;

const int SERIALIZATION_KERNEL_FUNCTION_RESULT_ID                                              = 102100;

const int SERIALIZATION_OUTLIER_DETECTION_MULTIVARIATE_RESULT_ID                               = 102200;
const int SERIALIZATION_OUTLIER_DETECTION_UNIVARIATE_RESULT_ID                                 = 102210;
const int SERIALIZATION_OUTLIER_DETECTION_BACON_RESULT_ID                                      = 102220;

const int SERIALIZATION_PIVOTED_QR_RESULT_ID                                                   = 102300;

const int SERIALIZATION_QR_RESULT_ID                                                           = 102400;
const int SERIALIZATION_QR_ONLINE_PARTIAL_RESULT_ID                                            = 102410;
const int SERIALIZATION_QR_DISTRIBUTED_PARTIAL_RESULT_ID                                       = 102420;
const int SERIALIZATION_QR_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID                                 = 102430;

const int SERIALIZATION_QUANTILES_RESULT_ID                                                    = 102500;

const int SERIALIZATION_WEAK_LEARNER_RESULT_ID                                                 = 102600;

const int SERIALIZATION_SVD_RESULT_ID                                                          = 102700;
const int SERIALIZATION_SVD_ONLINE_PARTIAL_RESULT_ID                                           = 102710;
const int SERIALIZATION_SVD_DISTRIBUTED_PARTIAL_RESULT_ID                                      = 102720;
const int SERIALIZATION_SVD_DISTRIBUTED_PARTIAL_RESULT_STEP3_ID                                = 102730;

const int SERIALIZATION_RELU_RESULT_ID                                                         = 103000;

const int SERIALIZATION_SORTING_RESULT_ID                                                      = 103100;

const int SERIALIZATION_SOFTMAX_RESULT_ID                                                      = 103200;
const int SERIALIZATION_LOGISTIC_RESULT_ID                                                     = 103300;
const int SERIALIZATION_TANH_RESULT_ID                                                         = 103400;

const int SERIALIZATION_SMOOTHRELU_RESULT_ID                                                   = 103500;
const int SERIALIZATION_ABS_RESULT_ID                                                          = 103600;

const int SERIALIZATION_OBJECTIVE_FUNCTION_RESULT_ID                                           = 103700;
const int SERIALIZATION_ITERATIVE_SOLVER_RESULT_ID                                             = 103810;
const int SERIALIZATION_ADAGRAD_RESULT_ID                                                      = 103820;
const int SERIALIZATION_LBFGS_RESULT_ID                                                        = 103830;
const int SERIALIZATION_SGD_RESULT_ID                                                          = 103840;

const int SERIALIZATION_NORMALIZATION_ZSCORE_RESULT_ID                                         = 103900;
const int SERIALIZATION_NORMALIZATION_MINMAX_RESULT_ID                                         = 103910;

const int SERIALIZATION_NEURAL_NETWORKS_TRAINING_MODEL_ID                                      = 104000;
const int SERIALIZATION_NEURAL_NETWORKS_PREDICTION_MODEL_ID                                    = 104010;
const int SERIALIZATION_NEURAL_NETWORKS_NUMERIC_TABLE_WEIGHTS_AND_BIASES_ID                    = 104020;
const int SERIALIZATION_NEURAL_NETWORKS_NUMERIC_TABLE_WEIGHTS_AND_BIASES_DERIVATIVES_ID        = 104030;
const int SERIALIZATION_NEURAL_NETWORKS_TENSOR_WEIGHTS_AND_BIASES_ID                           = 104040;
const int SERIALIZATION_NEURAL_NETWORKS_TENSOR_WEIGHTS_AND_BIASES_DERIVATIVES_ID               = 104050;

const int SERIALIZATION_NEURAL_NETWORKS_TRAINING_RESULT_ID                                     = 104100;
const int SERIALIZATION_NEURAL_NETWORKS_TRAINING_PARTIAL_RESULT_ID                             = 104101;
const int SERIALIZATION_NEURAL_NETWORKS_TRAINING_DISTRIBUTED_PARTIAL_RESULT_ID                 = 104102;
const int SERIALIZATION_NEURAL_NETWORKS_PREDICTION_RESULT_ID                                   = 104110;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_BACKWARD_RESULT_ID                              = 104120;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_FORWARD_RESULT_ID                               = 104130;

const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_ABS_BACKWARD_RESULT_ID                          = 104140;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_ABS_FORWARD_RESULT_ID                           = 104150;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOGISTIC_BACKWARD_RESULT_ID                     = 104160;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOGISTIC_FORWARD_RESULT_ID                      = 104170;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_RELU_BACKWARD_RESULT_ID                         = 104180;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_RELU_FORWARD_RESULT_ID                          = 104190;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SMOOTHRELU_BACKWARD_RESULT_ID                   = 104200;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SMOOTHRELU_FORWARD_RESULT_ID                    = 104210;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_TANH_BACKWARD_RESULT_ID                         = 104220;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_TANH_FORWARD_RESULT_ID                          = 104230;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_PRELU_FORWARD_RESULT_ID                         = 104240;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_PRELU_BACKWARD_RESULT_ID                        = 104250;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SOFTMAX_BACKWARD_RESULT_ID                      = 104260;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SOFTMAX_FORWARD_RESULT_ID                       = 104270;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_FULLYCONNECTED_BACKWARD_RESULT_ID               = 104320;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_FULLYCONNECTED_FORWARD_RESULT_ID                = 104330;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_DROPOUT_BACKWARD_RESULT_ID                      = 104340;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_DROPOUT_FORWARD_RESULT_ID                       = 104350;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_BATCH_NORMALIZATION_FORWARD_RESULT_ID           = 104360;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_BATCH_NORMALIZATION_BACKWARD_RESULT_ID          = 104370;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LRN_BACKWARD_RESULT_ID                          = 104380;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LRN_FORWARD_RESULT_ID                           = 104390;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPLIT_FORWARD_RESULT_ID                         = 104400;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPLIT_BACKWARD_RESULT_ID                        = 104410;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_CONVOLUTION2D_BACKWARD_RESULT_ID                = 104420;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_CONVOLUTION2D_FORWARD_RESULT_ID                 = 104430;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_CONCAT_BACKWARD_RESULT_ID                       = 104440;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_CONCAT_FORWARD_RESULT_ID                        = 104450;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_MAXIMUM_POOLING1D_FORWARD_RESULT_ID             = 104460;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_AVERAGE_POOLING1D_FORWARD_RESULT_ID             = 104470;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_MAXIMUM_POOLING1D_BACKWARD_RESULT_ID            = 104480;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_AVERAGE_POOLING1D_BACKWARD_RESULT_ID            = 104490;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_MAXIMUM_POOLING2D_FORWARD_RESULT_ID             = 104500;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_AVERAGE_POOLING2D_FORWARD_RESULT_ID             = 104510;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_MAXIMUM_POOLING2D_BACKWARD_RESULT_ID            = 104520;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_AVERAGE_POOLING2D_BACKWARD_RESULT_ID            = 104530;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_MAXIMUM_POOLING3D_FORWARD_RESULT_ID             = 104540;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_AVERAGE_POOLING3D_FORWARD_RESULT_ID             = 104550;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_MAXIMUM_POOLING3D_BACKWARD_RESULT_ID            = 104560;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_AVERAGE_POOLING3D_BACKWARD_RESULT_ID            = 104570;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOSS_SOFTMAX_CROSS_FORWARD_RESULT_ID            = 104580;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOSS_SOFTMAX_CROSS_BACKWARD_RESULT_ID           = 104590;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_STOCHASTIC_POOLING2D_FORWARD_RESULT_ID          = 104600;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_STOCHASTIC_POOLING2D_BACKWARD_RESULT_ID         = 104610;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOCALLYCONNECTED2D_FORWARD_RESULT_ID            = 104620;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOCALLYCONNECTED2D_BACKWARD_RESULT_ID           = 104625;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LCN_FORWARD_RESULT_ID                           = 104630;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LCN_BACKWARD_RESULT_ID                          = 104635;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_AVERAGE_POOLING2D_FORWARD_RESULT_ID     = 104640;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_AVERAGE_POOLING2D_BACKWARD_RESULT_ID    = 104650;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_MAXIMUM_POOLING2D_FORWARD_RESULT_ID     = 104660;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_MAXIMUM_POOLING2D_BACKWARD_RESULT_ID    = 104670;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_STOCHASTIC_POOLING2D_FORWARD_RESULT_ID  = 104680;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_STOCHASTIC_POOLING2D_BACKWARD_RESULT_ID = 104690;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_TRANSPOSED_CONV2D_BACKWARD_RESULT_ID            = 104700;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_TRANSPOSED_CONV2D_FORWARD_RESULT_ID             = 104710;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOSS_LOGISTIC_CROSS_FORWARD_RESULT_ID           = 104720;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LOSS_LOGISTIC_CROSS_BACKWARD_RESULT_ID          = 104730;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_RESHAPE_BACKWARD_RESULT_ID                      = 104740;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_RESHAPE_FORWARD_RESULT_ID                       = 104750;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_ELTWISE_SUM_FORWARD_RESULT_ID                   = 104760;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_ELTWISE_SUM_BACKWARD_RESULT_ID                  = 104770;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_ELU_FORWARD_RESULT_ID                           = 104780;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_ELU_BACKWARD_RESULT_ID                          = 104790;

const int SERIALIZATION_RIDGE_REGRESSION_MODELNORMEQ_ID                                        = 105000;
const int SERIALIZATION_RIDGE_REGRESSION_PARTIAL_RESULT_ID                                     = 105010;
const int SERIALIZATION_RIDGE_REGRESSION_TRAINING_RESULT_ID                                    = 105020;
const int SERIALIZATION_RIDGE_REGRESSION_PREDICTION_RESULT_ID                                  = 105030;

const int SERIALIZATION_K_NEAREST_NEIGHBOR_MODEL_ID                                            = 106000;
const int SERIALIZATION_K_NEAREST_NEIGHBOR_TRAINING_RESULT_ID                                  = 106010;

const int SERIALIZATION_DECISION_FOREST_CLASSIFICATION_MODEL_ID                                = 107000;
const int SERIALIZATION_DECISION_FOREST_CLASSIFICATION_TRAINING_RESULT_ID                      = 107010;
const int SERIALIZATION_DECISION_FOREST_CLASSIFICATION_PREDICTION_RESULT_ID                    = 107020;
const int SERIALIZATION_DECISION_FOREST_REGRESSION_MODEL_ID                                    = 107030;
const int SERIALIZATION_DECISION_FOREST_REGRESSION_TRAINING_RESULT_ID                          = 107040;
const int SERIALIZATION_DECISION_FOREST_REGRESSION_PREDICTION_RESULT_ID                        = 107050;
const int SERIALIZATION_GBT_CLASSIFICATION_MODEL_ID                                            = 107100;
const int SERIALIZATION_GBT_CLASSIFICATION_TRAINING_RESULT_ID                                  = 107110;
const int SERIALIZATION_GBT_CLASSIFICATION_PREDICTION_RESULT_ID                                = 107120;
const int SERIALIZATION_GBT_REGRESSION_MODEL_ID                                                = 107130;
const int SERIALIZATION_GBT_REGRESSION_TRAINING_RESULT_ID                                      = 107140;
const int SERIALIZATION_GBT_REGRESSION_PREDICTION_RESULT_ID                                    = 107150;
const int SERIALIZATION_GBT_DECISION_TREE_ID                                                   = 107160;

const int SERIALIZATION_DECISION_TREE_CLASSIFICATION_MODEL_ID                                  = 108000;
const int SERIALIZATION_DECISION_TREE_CLASSIFICATION_TRAINING_RESULT_ID                        = 108010;
const int SERIALIZATION_DECISION_TREE_REGRESSION_MODEL_ID                                      = 108020;
const int SERIALIZATION_DECISION_TREE_REGRESSION_TRAINING_RESULT_ID                            = 108030;
const int SERIALIZATION_DECISION_TREE_REGRESSION_PREDICTION_RESULT_ID                          = 108040;

const int SERIALIZATION_REGRESSION_TRAINING_RESULT_ID                                          = 109000;
const int SERIALIZATION_REGRESSION_PREDICTION_RESULT_ID                                        = 109020;

const int SERIALIZATION_LM_TRAINING_RESULT_ID                                                  = 109100;
const int SERIALIZATION_LM_PREDICTION_RESULT_ID                                                = 109120;

const int SERIALIZATION_LOGISTIC_REGRESSION_MODEL_ID                                           = 110000;
const int SERIALIZATION_LOGISTIC_REGRESSION_TRAINING_RESULT_ID                                 = 110010;
const int SERIALIZATION_LOGISTIC_REGRESSION_PREDICTION_RESULT_ID                               = 110020;
};

#define DAAL_NEW_DELETE()                                \
static void *operator new(std::size_t sz)                \
{                                                        \
    return daal::services::daal_malloc(sz);              \
}                                                        \
static void *operator new[](std::size_t sz)              \
{                                                        \
    return daal::services::daal_malloc(sz);              \
}                                                        \
static void *operator new(std::size_t sz, void *where)   \
{                                                        \
    return where;                                        \
}                                                        \
static void *operator new[](std::size_t sz, void *where) \
{                                                        \
    return where;                                        \
}                                                        \
static void operator delete(void *ptr, std::size_t sz)   \
{                                                        \
    daal::services::daal_free(ptr);                      \
}                                                        \
static void operator delete[](void *ptr, std::size_t sz) \
{                                                        \
    daal::services::daal_free(ptr);                      \
}

#define DAAL_CAST_OPERATOR(ClassName)                                           \
template<class U>                                                               \
static services::SharedPtr<ClassName> cast(const services::SharedPtr<U> &r)     \
{                                                                               \
    return services::dynamicPointerCast<ClassName, U>(r);                       \
}

#define DAAL_DOWN_CAST_OPERATOR(DstClassName, SrcClassName)                                     \
static services::SharedPtr<DstClassName> downCast(const services::SharedPtr<SrcClassName> &r)   \
{                                                                                               \
    return services::dynamicPointerCast<DstClassName, SrcClassName>(r);                         \
}

#ifndef DAAL_ALGORITHM_FP_TYPE
#define DAAL_ALGORITHM_FP_TYPE       float /* default type for algorithms */
#endif
#ifndef DAAL_DATA_TYPE
#define DAAL_DATA_TYPE               float /* default type for tables and tensors */
#endif
#ifndef DAAL_SUMMARY_STATISTICS_TYPE
#define DAAL_SUMMARY_STATISTICS_TYPE float /* default type for summary statistics in data source */
#endif

#ifdef DEBUG_ASSERT
    #include <assert.h>
    #define DAAL_ASSERT(cond) assert(cond);
#else
    #define DAAL_ASSERT(cond)
#endif

#define DAAL_CHECK(cond, error) if(!(cond)) return services::Status(error);
#define DAAL_CHECK_EX(cond, error, detailType, detailValue) if(!(cond)) return services::Status(services::Error::create(error, detailType, detailValue));
#define DAAL_CHECK_THR(cond, error) {using namespace services; if(!(cond)) { safeStat.add(error); return; } }

#define DAAL_CHECK_MALLOC(cond) DAAL_CHECK(cond, services::ErrorMemoryAllocationFailed)
#define DAAL_CHECK_MALLOC_THR(cond) DAAL_CHECK_THR(cond, services::ErrorMemoryAllocationFailed)

#define DAAL_CHECK_STATUS(destVar, srcVal) { destVar |= (srcVal); if(!(destVar)) return destVar; }
#define DAAL_CHECK_STATUS_VAR(statVal)     { if(!(statVal)) return statVal; }
#define DAAL_CHECK_STATUS_THR(statVal)     { if(!(statVal)) { safeStat.add(statVal); return; } }
#define DAAL_CHECK_SAFE_STATUS()           { if(!(safeStat)) return safeStat.detach(); }

#define DAAL_CHECK_BLOCK_STATUS(block)     { if(!(block).status()) return (block).status(); }
#define DAAL_CHECK_BLOCK_STATUS_THR(block) DAAL_CHECK_STATUS_THR((block).status())

#define DAAL_DEFAULT_CREATE_IMPL(Type)                              \
{                                                                   \
    services::Status defaultSt;                                     \
    services::Status &st = (stat ? *stat : defaultSt);              \
    services::SharedPtr<Type> result(new Type(st));                 \
    if (!result) { st.add(services::ErrorMemoryAllocationFailed); } \
    if (!st) { result.reset(); }                                    \
    return result;                                                  \
}

#define DAAL_DEFAULT_CREATE_IMPL_EX(Type, ...)                      \
{                                                                   \
    services::Status defaultSt;                                     \
    services::Status &st = (stat ? *stat : defaultSt);              \
    services::SharedPtr<Type> result(new Type(__VA_ARGS__, st));    \
    if (!result) { st.add(services::ErrorMemoryAllocationFailed); } \
    if (!st) { result.reset(); }                                    \
    return result;                                                  \
}

#define DAAL_TEMPLATE_ARGUMENTS(...) __VA_ARGS__

#define DAAL_DEFAULT_CREATE_TEMPLATE_IMPL(Type, TemplateArgs)                       \
{                                                                                   \
    services::Status defaultSt;                                                     \
    services::Status &st = (stat ? *stat : defaultSt);                              \
    services::SharedPtr<Type<TemplateArgs> > result(new Type<TemplateArgs>(st));    \
    if (!result) { st.add(services::ErrorMemoryAllocationFailed); }                 \
    if (!st) { result.reset(); }                                                    \
    return result;                                                                  \
}

#define DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(Type, TemplateArgs, ...)                           \
{                                                                                               \
    services::Status defaultSt;                                                                 \
    services::Status &st = (stat ? *stat : defaultSt);                                          \
    services::SharedPtr<Type<TemplateArgs> > result(new Type<TemplateArgs>(__VA_ARGS__, st));   \
    if (!result) { st.add(services::ErrorMemoryAllocationFailed); }                             \
    if (!st) { result.reset(); }                                                                \
    return result;                                                                              \
}

#define DAAL_ALLOCATE_TENSOR_AND_SET(s, collectionId, tensorDim)                                                                        \
{                                                                                                                                       \
    set(collectionId, data_management::HomogenTensor<algorithmFPType>::create(tensorDim, data_management::Tensor::doAllocate, &s));     \
    DAAL_CHECK_STATUS_VAR(s);                                                                                                           \
}

#define DAAL_CHECK_NUMERIC_TABLE(destVar, ...) DAAL_CHECK_STATUS(destVar, data_management::checkNumericTable(__VA_ARGS__))
#define DAAL_CHECK_TENSOR(destVar, ...) DAAL_CHECK_STATUS(destVar, data_management::checkTensor(__VA_ARGS__))

#endif

/* file: daal_defines.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
    step1Local     = 0,        /*!< First  step of the distributed processing mode */
    step2Master    = 1,        /*!< Second step of the distributed processing mode */
    step3Local     = 2,        /*!< Third  step of the distributed processing mode */
    step4Local     = 3        /*!< Fourth step of the distributed processing mode */
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
template<class U, class V> struct IsSameType      { static const bool value     = false; };

/**
 * <a name="DAAL-STRUCT-ISSAMETYPE"></a>
 * Indicates if the types are equal
 */
template<class U>          struct IsSameType<U, U> { static const bool value     = true; };

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
const int SERIALIZATION_JAVANIOCSR_NT_ID                                                       = 9000;
const int SERIALIZATION_JAVANIO_NT_ID                                                          = 10000;
const int SERIALIZATION_PACKEDSYMMETRIC_NT_ID                                                  = 11000;
const int SERIALIZATION_PACKEDTRIANGULAR_NT_ID                                                 = 12000;
const int SERIALIZATION_MERGE_NT_ID                                                            = 13000;

const int SERIALIZATION_HOMOGEN_TENSOR_ID                                                      = 20000;
const int SERIALIZATION_JAVANIO_TENSOR_ID                                                      = 21000;

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

const int SERIALIZATION_SVM_MODEL_ID                                                           = 100800;
const int SERIALIZATION_SVM_TRAINING_RESULT_ID                                                 = 100810;

const int SERIALIZATION_MULTI_CLASS_CLASSIFIER_MODEL_ID                                        = 100900;
const int SERIALIZATION_MULTICLASS_CLASSIFIER_RESULT_ID                                        = 100910;

const int SERIALIZATION_COVARIANCE_PARTIAL_RESULT_ID                                           = 101000;
const int SERIALIZATION_COVARIANCE_RESULT_ID                                                   = 101010;

const int SERIALIZATION_KMEANS_PARTIAL_RESULT_ID                                               = 101100;
const int SERIALIZATION_KMEANS_RESULT_ID                                                       = 101110;
const int SERIALIZATION_KMEANS_INIT_PARTIAL_RESULT_ID                                          = 101200;
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
const int SERIALIZATION_IMPLICIT_ALS_TRAINING_INIT_PARTIAL_RESULT_ID                           = 101650;
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

const int SERIALIZATION_NORMALIZATION_ZSCORE_RESULT_ID                                         = 103900;

const int SERIALIZATION_NEURAL_NETWORKS_TRAINING_MODEL_ID                                      = 104000;
const int SERIALIZATION_NEURAL_NETWORKS_PREDICTION_MODEL_ID                                    = 104010;
const int SERIALIZATION_NEURAL_NETWORKS_NUMERIC_TABLE_WEIGHTS_AND_BIASES_ID                    = 104020;
const int SERIALIZATION_NEURAL_NETWORKS_NUMERIC_TABLE_WEIGHTS_AND_BIASES_DERIVATIVES_ID        = 104030;
const int SERIALIZATION_NEURAL_NETWORKS_TENSOR_WEIGHTS_AND_BIASES_ID                           = 104040;
const int SERIALIZATION_NEURAL_NETWORKS_TENSOR_WEIGHTS_AND_BIASES_DERIVATIVES_ID               = 104050;

const int SERIALIZATION_NEURAL_NETWORKS_TRAINING_RESULT_ID                                     = 104100;
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
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_LCN_FORWARD_RESULT_ID                           = 104630;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_AVERAGE_POOLING2D_FORWARD_RESULT_ID     = 104640;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_AVERAGE_POOLING2D_BACKWARD_RESULT_ID    = 104650;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_MAXIMUM_POOLING2D_FORWARD_RESULT_ID     = 104660;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_MAXIMUM_POOLING2D_BACKWARD_RESULT_ID    = 104670;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_STOCHASTIC_POOLING2D_FORWARD_RESULT_ID  = 104680;
const int SERIALIZATION_NEURAL_NETWORKS_LAYERS_SPATIAL_STOCHASTIC_POOLING2D_BACKWARD_RESULT_ID = 104690;
const int SERIALIZATION_RIDGE_REGRESSION_MODELNORMEQ_ID                                        = 105000;
const int SERIALIZATION_RIDGE_REGRESSION_PARTIAL_RESULT_ID                                     = 105010;
const int SERIALIZATION_RIDGE_REGRESSION_TRAINING_RESULT_ID                                    = 105020;
const int SERIALIZATION_RIDGE_REGRESSION_PREDICTION_RESULT_ID                                  = 105030;


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

#define DAAL_CHECK(cond, error)          \
if(!(cond))                              \
{                                        \
    this->_errors->add(services::error); \
    return;                              \
}

#define DAAL_CHECK_EX(cond, error, detailType, detailValue)                      \
if(!(cond))                                                                      \
{                                                                                \
    this->_errors->add(services::Error::create(error, detailType, detailValue)); \
    return;                                                                      \
}

#define DAAL_ALLOCATE_TENSOR_AND_SET(collectionId, tensorDim)                                                                \
    set(collectionId, services::SharedPtr<data_management::HomogenTensor<algorithmFPType> >(                                 \
                      new data_management::HomogenTensor<algorithmFPType>(tensorDim, data_management::Tensor::doAllocate))); \

#endif

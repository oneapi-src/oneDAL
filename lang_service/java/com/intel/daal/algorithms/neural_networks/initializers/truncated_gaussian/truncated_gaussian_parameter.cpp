/* file: truncated_gaussian_parameter.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

#include <jni.h>
#include "neural_networks/initializers/truncated_gaussian/JTruncatedGaussianParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianParameter
 * Method:    cSetA
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianParameter_cSetA
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble a, jint prec)
{
    if(prec == 0)
    {
        (((initializers::truncated_gaussian::Parameter<double> *)cParameter))->a = (double)a;
    }
    else
    {
        (((initializers::truncated_gaussian::Parameter<float> *)cParameter))->a = (double)a;
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianParameter
 * Method:    cSetB
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianParameter_cSetB
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble b, jint prec)
{
    if(prec == 0)
    {
        (((initializers::truncated_gaussian::Parameter<double> *)cParameter))->b = (double)b;
    }
    else
    {
        (((initializers::truncated_gaussian::Parameter<float> *)cParameter))->b = (double)b;
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianParameter
 * Method:    cGetA
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianParameter_cGetA
  (JNIEnv *env, jobject thisObj, jlong cParameter, jint prec)
{
    if(prec == 0)
    {
        return (jdouble)((((initializers::truncated_gaussian::Parameter<double> *)cParameter))->a);
    }
    else
    {
        return (jdouble)((((initializers::truncated_gaussian::Parameter<float> *)cParameter))->a);
    }
    return 0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianParameter
 * Method:    cGetB
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianParameter_cGetB
  (JNIEnv *env, jobject thisObj, jlong cParameter, jint prec)
{
    if(prec == 0)
    {
        return (jdouble)((((initializers::truncated_gaussian::Parameter<double> *)cParameter))->b);
    }
    else
    {
        return (jdouble)((((initializers::truncated_gaussian::Parameter<float> *)cParameter))->b);
    }
    return 0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianParameter
 * Method:    cSetMean
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianParameter_cSetMean
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble mean, jint prec)
{
    if(prec == 0)
    {
        (((initializers::truncated_gaussian::Parameter<double> *)cParameter))->mean = (double)mean;
    }
    else
    {
        (((initializers::truncated_gaussian::Parameter<float> *)cParameter))->mean = (double)mean;
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianParameter
 * Method:    cSetSigma
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianParameter_cSetSigma
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble sigma, jint prec)
{
    if(prec == 0)
    {
        (((initializers::truncated_gaussian::Parameter<double> *)cParameter))->sigma = (double)sigma;
    }
    else
    {
        (((initializers::truncated_gaussian::Parameter<float> *)cParameter))->sigma = (double)sigma;
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianParameter
 * Method:    cGetMean
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianParameter_cGetMean
  (JNIEnv *env, jobject thisObj, jlong cParameter, jint prec)
{
    if(prec == 0)
    {
        return (jdouble)((((initializers::truncated_gaussian::Parameter<double> *)cParameter))->mean);
    }
    else
    {
        return (jdouble)((((initializers::truncated_gaussian::Parameter<float> *)cParameter))->mean);
    }
    return 0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_truncated_gaussian_TruncatedGaussianParameter
 * Method:    cGetSigma
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianParameter_cGetSigma
  (JNIEnv *env, jobject thisObj, jlong cParameter, jint prec)
{
    if(prec == 0)
    {
        return (jdouble)((((initializers::truncated_gaussian::Parameter<double> *)cParameter))->sigma);
    }
    else
    {
        return (jdouble)((((initializers::truncated_gaussian::Parameter<float> *)cParameter))->sigma);
    }
    return 0;
}

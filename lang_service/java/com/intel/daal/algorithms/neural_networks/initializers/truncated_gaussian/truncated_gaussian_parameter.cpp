/* file: truncated_gaussian_parameter.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
 * Method:    cSetSeed
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianParameter_cSetSeed
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong seed, jint prec)
{
    if(prec == 0)
    {
        (((initializers::truncated_gaussian::Parameter<double> *)cParameter))->seed = (size_t)seed;
    }
    else
    {
        (((initializers::truncated_gaussian::Parameter<float> *)cParameter))->seed = (size_t)seed;
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
 * Method:    cGetSeed
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_truncated_1gaussian_TruncatedGaussianParameter_cGetSeed
  (JNIEnv *env, jobject thisObj, jlong cParameter, jint prec)
{
    if(prec == 0)
    {
        return (jlong)((((initializers::truncated_gaussian::Parameter<double> *)cParameter))->seed);
    }
    else
    {
        return (jlong)((((initializers::truncated_gaussian::Parameter<float> *)cParameter))->seed);
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

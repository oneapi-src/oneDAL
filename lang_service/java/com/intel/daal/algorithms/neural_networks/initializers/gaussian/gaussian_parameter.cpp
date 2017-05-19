/* file: gaussian_parameter.cpp */
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
#include "neural_networks/initializers/gaussian/JGaussianParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_gaussian_GaussianParameter
 * Method:    cSetA
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_gaussian_GaussianParameter_cSetA
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble a)
{
    (((initializers::gaussian::Parameter *)cParameter))->a = (double)a;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_gaussian_GaussianParameter
 * Method:    cSetSigma
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_gaussian_GaussianParameter_cSetSigma
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble sigma)
{
    (((initializers::gaussian::Parameter *)cParameter))->sigma = (double)sigma;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_gaussian_GaussianParameter
 * Method:    cSetSeed
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_gaussian_GaussianParameter_cSetSeed
  (JNIEnv *env, jobject thisObj, jlong cParameter, jlong seed)
{
    (((initializers::gaussian::Parameter *)cParameter))->seed = (size_t)seed;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_gaussian_GaussianParameter
 * Method:    cGetA
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_gaussian_GaussianParameter_cGetA
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jdouble)((((initializers::gaussian::Parameter *)cParameter))->a);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_gaussian_GaussianParameter
 * Method:    cGetSigma
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_gaussian_GaussianParameter_cGetSigma
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jdouble)((((initializers::gaussian::Parameter *)cParameter))->sigma);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_gaussian_GaussianParameter
 * Method:    cGetSeed
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_gaussian_GaussianParameter_cGetSeed
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jlong)((((initializers::gaussian::Parameter *)cParameter))->seed);
}

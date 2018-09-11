/* file: gaussian_parameter.cpp */
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

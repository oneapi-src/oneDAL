/* file: parameter.cpp */
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
#include "distributions/normal/JParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_distributions_normal_Parameter
 * Method:    cSetA
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_distributions_normal_Parameter_cSetA
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble a)
{
    (((distributions::normal::Parameter<float> *)cParameter))->a = (float)a;
}

/*
 * Class:     com_intel_daal_algorithms_distributions_normal_Parameter
 * Method:    cSetSigma
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_distributions_normal_Parameter_cSetSigma
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble sigma)
{
    (((distributions::normal::Parameter<float> *)cParameter))->sigma = (float)sigma;
}

/*
 * Class:     com_intel_daal_algorithms_distributions_normal_Parameter
 * Method:    cGetA
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_distributions_normal_Parameter_cGetA
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jdouble)((((distributions::normal::Parameter<float> *)cParameter))->a);
}

/*
 * Class:     com_intel_daal_algorithms_distributions_normal_Parameter
 * Method:    cGetSigma
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_distributions_normal_Parameter_cGetSigma
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jdouble)((((distributions::normal::Parameter<float> *)cParameter))->sigma);
}

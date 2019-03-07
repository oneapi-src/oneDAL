/* file: parameter.cpp */
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
#include "distributions/bernoulli/JParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_distributions_bernoulli_Parameter
 * Method:    cSetP
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_distributions_bernoulli_Parameter_cSetP
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble p)
{
    (((distributions::bernoulli::Parameter<float> *)cParameter))->p = (double)p;
}

/*
 * Class:     com_intel_daal_algorithms_distributions_bernoulli_Parameter
 * Method:    cGetP
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_distributions_bernoulli_Parameter_cGetP
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jdouble)((((distributions::bernoulli::Parameter<float> *)cParameter))->p);
}

/* file: uniform_parameter.cpp */
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
#include "neural_networks/initializers/uniform/JUniformParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_uniform_UniformParameter
 * Method:    cSetA
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_uniform_UniformParameter_cSetA
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble a)
{
    (((initializers::uniform::Parameter *)cParameter))->a = (double)a;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_uniform_UniformParameter
 * Method:    cSetB
 * Signature: (JD)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_uniform_UniformParameter_cSetB
  (JNIEnv *env, jobject thisObj, jlong cParameter, jdouble b)
{
    (((initializers::uniform::Parameter *)cParameter))->b = (double)b;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_uniform_UniformParameter
 * Method:    cGetA
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_uniform_UniformParameter_cGetA
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jdouble)((((initializers::uniform::Parameter *)cParameter))->a);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_uniform_UniformParameter
 * Method:    cGetB
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_uniform_UniformParameter_cGetB
  (JNIEnv *env, jobject thisObj, jlong cParameter)
{
    return (jdouble)((((initializers::uniform::Parameter *)cParameter))->b);
}

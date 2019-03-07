/* file: xavier_batch.cpp */
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
#include "neural_networks/initializers/xavier/JXavierBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_xavier_XavierBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_xavier_XavierBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<initializers::xavier::Method, initializers::xavier::Batch, initializers::xavier::defaultDense>::newObj(
               prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_xavier_XavierBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_xavier_XavierBatch_cInitParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<initializers::xavier::Method, initializers::xavier::Batch, initializers::xavier::defaultDense>::getParameter(
        prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_xavier_XavierBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_xavier_XavierBatch_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<initializers::xavier::Method, initializers::xavier::Batch, initializers::xavier::defaultDense>::getResult(
        prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_initializers_xavier_XavierBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_initializers_xavier_XavierBatch_cClone
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<initializers::xavier::Method, initializers::xavier::Batch, initializers::xavier::defaultDense>::getClone(
        prec, method, algAddr);
}

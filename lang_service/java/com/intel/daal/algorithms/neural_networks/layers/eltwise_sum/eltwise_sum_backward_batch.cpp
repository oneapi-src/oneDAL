/* file: eltwise_sum_backward_batch.cpp */
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
#include "neural_networks/layers/eltwise_sum/JEltwiseSumBackwardBatch.h"

#include "daal.h"
#include "common_helpers.h"

using namespace daal;
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardBatch
 * Method:    cInit
 * Signature: (II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardBatch_cInit
  (JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<eltwise_sum::Method, eltwise_sum::backward::Batch, eltwise_sum::defaultDense>::
        newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardBatch_cGetInput
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<eltwise_sum::Method, eltwise_sum::backward::Batch, eltwise_sum::defaultDense>::
        getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardBatch_cInitParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<eltwise_sum::Method, eltwise_sum::backward::Batch, eltwise_sum::defaultDense>::
        getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardBatch_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<eltwise_sum::Method, eltwise_sum::backward::Batch, eltwise_sum::defaultDense>::
        getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardBatch
 * Method:    cSetResult
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardBatch_cSetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jlong resAddr)
{
    jniBatch<eltwise_sum::Method, eltwise_sum::backward::Batch, eltwise_sum::defaultDense>::
        setResult<eltwise_sum::backward::Result>(prec, method, algAddr, resAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumBackwardBatch
 * Method:    cClone
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumBackwardBatch_cClone
  (JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<eltwise_sum::Method, eltwise_sum::backward::Batch, eltwise_sum::defaultDense>::
        getClone(prec, method, algAddr);
}

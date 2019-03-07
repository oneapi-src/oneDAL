/* file: loss_backward_batch.cpp */
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
#include "neural_networks/layers/loss/JLossBackwardBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_loss_LossBackwardBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_loss_LossBackwardBatch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return (jlong) (*((SharedPtr<loss::backward::Batch> *)algAddr))->getLayerInput();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_loss_LossBackwardBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_loss_LossBackwardBatch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SerializationIfacePtr *ptr = new SerializationIfacePtr;
    *ptr = (*((SharedPtr<loss::backward::Batch> *)algAddr))->getLayerResult();
    return (jlong)ptr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_loss_LossBackwardBatch
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_BackwardLayer_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(SharedPtr<loss::backward::Batch> *)addr;
}

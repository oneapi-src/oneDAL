/* file: loss_forward_batch.cpp */
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
#include "neural_networks/layers/loss/JLossForwardBatch.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_loss_LossForwardBatch
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_loss_LossForwardBatch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return (jlong) (*((SharedPtr<loss::forward::Batch> *)algAddr))->getLayerInput();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_loss_LossForwardBatch
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_loss_LossForwardBatch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SerializationIfacePtr *ptr = new SerializationIfacePtr;
    *ptr = (*((SharedPtr<loss::forward::Batch> *)algAddr))->getLayerResult();
    return (jlong)ptr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_loss_LossForwardBatch
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardLayer_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(SharedPtr<loss::forward::Batch> *)addr;
}

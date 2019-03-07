/* file: eltwise_sum_forward_input.cpp */
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
#include "neural_networks/layers/JForwardInputLayerDataId.h"
#include "neural_networks/layers/eltwise_sum/JEltwiseSumForwardInput.h"

#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers::eltwise_sum;
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumForwardInput
 * Method:    cSetLayerDataTensor
 * Signature: (JIJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumForwardInput_cSetLayerDataTensor
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong tensorAddr, jlong index)
{
    jniInput<forward::Input>::set<layers::forward::InputLayerDataId, Tensor>(inputAddr, id, tensorAddr, (size_t)index);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumForwardInput
 * Method:    cSetEltwiseInputTensor
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumForwardInput_cSetEltwiseInputTensor
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong tensorAddr)
{
    jniInput<forward::Input>::set<forward::InputId, Tensor>(inputAddr, id, tensorAddr);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumForwardInput
 * Method:    cGetLayerDataTensor
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumForwardInput_cGetLayerDataTensor
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong index)
{
    return jniInput<forward::Input>::get<layers::forward::InputLayerDataId, Tensor>(inputAddr, id, (size_t)index);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_eltwise_sum_EltwiseSumForwardInput
 * Method:    cGetLayerDataTensor
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_eltwise_1sum_EltwiseSumForwardInput_cGetEltwiseInputTensor
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    return jniInput<forward::Input>::get<forward::InputId, Tensor>(inputAddr, id);
}

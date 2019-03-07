/* file: concat_forward_input.cpp */
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
#include "neural_networks/layers/concat/JConcatForwardInput.h"

#include "daal.h"

#include "common_helpers.h"

#define inputLayerDataId com_intel_daal_algorithms_neural_networks_layers_ForwardInputLayerDataId_inputLayerDataId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers::concat;
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_concat_ConcatForwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_concat_ConcatForwardInput_cSetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr, jlong index)
{
    if (id == inputLayerDataId)
    {
        jniInput<forward::Input>::set<layers::forward::InputLayerDataId, Tensor>(inputAddr, id, ntAddr, (size_t)index);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_concat_ConcatForwardInput
 * Method:    cGetInput
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_concat_ConcatForwardInput_cGetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong index)
{
    if (id == inputLayerDataId)
    {
        return (jlong)jniInput<forward::Input>::get<layers::forward::InputLayerDataId, Tensor>(inputAddr, id, (size_t)index);
    }

    return (jlong)0;
}

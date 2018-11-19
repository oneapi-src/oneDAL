/* file: backward_input.cpp */
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
#include "neural_networks/layers/JBackwardInput.h"
#include "neural_networks/layers/JBackwardInputId.h"
#include "neural_networks/layers/JBackwardInputLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define inputGradientId com_intel_daal_algorithms_neural_networks_layers_BackwardInputId_inputGradientId
#define inputFromForwardId com_intel_daal_algorithms_neural_networks_layers_BackwardInputLayerDataId_inputFromForwardId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_BackwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_BackwardInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == inputGradientId)
    {
        jniInput<backward::Input>::set<backward::InputId, Tensor>(inputAddr, id, ntAddr);
    } else if (id == inputFromForwardId)
    {
        jniInput<backward::Input>::set<backward::InputLayerDataId, KeyValueDataCollection>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_BackwardInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_BackwardInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == inputGradientId)
    {
        return jniInput<backward::Input>::get<backward::InputId, Tensor>(inputAddr, id);
    } else if (id == inputFromForwardId)
    {
        return jniInput<backward::Input>::get<backward::InputLayerDataId, KeyValueDataCollection>(inputAddr, id);
    }

    return (jlong)0;
}

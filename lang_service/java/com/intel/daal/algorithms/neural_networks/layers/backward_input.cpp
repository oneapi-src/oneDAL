/* file: backward_input.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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

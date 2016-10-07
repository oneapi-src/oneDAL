/* file: concat_forward_input.cpp */
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

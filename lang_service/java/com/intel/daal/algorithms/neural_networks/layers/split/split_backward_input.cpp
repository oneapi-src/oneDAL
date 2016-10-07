/* file: split_backward_input.cpp */
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
#include "neural_networks/layers/split/JSplitBackwardInput.h"
#include "neural_networks/layers/split/JSplitBackwardInputLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define inputGradientCollectionId com_intel_daal_algorithms_neural_networks_layers_split_SplitBackwardInputLayerDataId_inputGradientCollectionId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers::split;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitBackwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitBackwardInput_cSetInput__JIJ
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == inputGradientCollectionId)
    {
        jniInput<backward::Input>::set<backward::InputLayerDataId, KeyValueDataCollection>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitBackwardInput
 * Method:    cSetInput
 * Signature: (JIJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitBackwardInput_cSetInput__JIJJ
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr, jlong index)
{
    if (id == inputGradientCollectionId)
    {
        jniInput<backward::Input>::set<backward::InputLayerDataId, Tensor>(inputAddr, id, ntAddr, (size_t)index);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitBackwardInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitBackwardInput_cGetInput__JI
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == inputGradientCollectionId)
    {
        return jniInput<backward::Input>::get<backward::InputLayerDataId, KeyValueDataCollection>(inputAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitBackwardInput
 * Method:    cGetInput
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitBackwardInput_cGetInput__JIJ
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong index)
{
    if (id == inputGradientCollectionId)
    {
        return jniInput<backward::Input>::get<backward::InputLayerDataId, Tensor>(inputAddr, id, (size_t)index);
    }

    return (jlong)0;
}

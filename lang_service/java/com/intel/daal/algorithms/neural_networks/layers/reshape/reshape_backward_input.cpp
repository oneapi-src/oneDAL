/* file: reshape_backward_input.cpp */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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
#include "neural_networks/layers/reshape/JReshapeBackwardInput.h"
#include "neural_networks/layers/reshape/JReshapeLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define auxDataId com_intel_daal_algorithms_neural_networks_layers_reshape_ReshapeLayerDataId_auxDataId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers::reshape;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_reshape_ReshapeBackwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_reshape_ReshapeBackwardInput_cSetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == auxDataId)
    {
        jniInput<backward::Input>::set<LayerDataId, NumericTable>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_reshape_ReshapeBackwardInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_reshape_ReshapeBackwardInput_cGetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == auxDataId)
    {
        return jniInput<backward::Input>::get<LayerDataId, NumericTable>(inputAddr, id);
    }

    return (jlong)0;
}

/* file: forward_result.cpp */
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
#include "neural_networks/layers/JForwardResult.h"
#include "neural_networks/layers/JForwardResultId.h"
#include "neural_networks/layers/JForwardResultLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define valueId com_intel_daal_algorithms_neural_networks_layers_ForwardResultId_valueId
#define resultForBackwardId com_intel_daal_algorithms_neural_networks_layers_ForwardResultLayerDataId_resultForBackwardId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ForwardResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardResult_cGetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == valueId)
    {
        return jniArgument<forward::Result>::get<forward::ResultId, Tensor>(resAddr, id);
    } else if (id == resultForBackwardId)
    {
        return jniArgument<forward::Result>::get<forward::ResultLayerDataId, KeyValueDataCollection>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ForwardResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    if (id == valueId)
    {
        jniArgument<forward::Result>::set<forward::ResultId, Tensor>(resAddr, id, ntAddr);
    } else if (id == resultForBackwardId)
    {
        jniArgument<forward::Result>::set<forward::ResultLayerDataId, KeyValueDataCollection>(resAddr, id, ntAddr);
    }
}

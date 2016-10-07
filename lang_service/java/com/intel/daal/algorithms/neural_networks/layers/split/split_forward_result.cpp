/* file: split_forward_result.cpp */
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
#include "neural_networks/layers/split/JSplitForwardResult.h"
#include "neural_networks/layers/split/JSplitForwardResultLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define valueCollectionId com_intel_daal_algorithms_neural_networks_layers_split_SplitForwardResultLayerDataId_valueCollectionId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers::split;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitForwardResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitForwardResult_cNewResult
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<forward::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitForwardResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitForwardResult_cGetValue__JI
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == valueCollectionId)
    {
        return jniArgument<forward::Result>::get<forward::ResultLayerDataId, KeyValueDataCollection>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitForwardResult
 * Method:    cGetValue
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitForwardResult_cGetValue__JIJ
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong index)
{
    if (id == valueCollectionId)
    {
        return jniArgument<forward::Result>::get<forward::ResultLayerDataId, Tensor>(resAddr, id, (size_t)index);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitForwardResult
 * Method:    cSetValue
 * Signature: (JIJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitForwardResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr, jlong index)
{
    if (id == valueCollectionId)
    {
        jniArgument<forward::Result>::set<forward::ResultLayerDataId, Tensor>(resAddr, id, ntAddr, (size_t)index);
    }
}

/* file: forward_layers.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "com_intel_daal_algorithms_neural_networks_ForwardLayers.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_ForwardLayers
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_ForwardLayers_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new SharedPtr<Collection<layers::forward::LayerIfacePtr > >(new Collection<layers::forward::LayerIfacePtr >()));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_ForwardLayers
 * Method:    cSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_ForwardLayers_cSize
(JNIEnv *env, jobject thisObj, jlong colAddr)
{
    return (*(SharedPtr<Collection<layers::forward::LayerIfacePtr > >*)colAddr)->size();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_ForwardLayers
 * Method:    cGet
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_ForwardLayers_cGet
(JNIEnv *env, jobject thisObj, jlong colAddr, jlong index)
{
    return (jlong) (new layers::forward::LayerIfacePtr
        ((*(SharedPtr<Collection<layers::forward::LayerIfacePtr > >*)colAddr)->get((size_t)index)));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_ForwardLayers
 * Method:    cPushBack
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_ForwardLayers_cPushBack
(JNIEnv *env, jobject thisObj, jlong colAddr, jlong layerAddr)
{
    (*(SharedPtr<Collection<layers::forward::LayerIfacePtr > >*)colAddr)->
        push_back(*((layers::forward::LayerIfacePtr *)layerAddr));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_ForwardLayers
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_ForwardLayers_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(SharedPtr<Collection<layers::forward::LayerIfacePtr > > *)addr;
}

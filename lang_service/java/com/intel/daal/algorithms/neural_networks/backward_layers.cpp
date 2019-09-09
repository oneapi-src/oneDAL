/* file: backward_layers.cpp */
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
#include "com_intel_daal_algorithms_neural_networks_BackwardLayers.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_BackwardLayers
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_BackwardLayers_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new SharedPtr<Collection<layers::backward::LayerIfacePtr > >(new Collection<layers::backward::LayerIfacePtr >()));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_BackwardLayers
 * Method:    cSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_BackwardLayers_cSize
(JNIEnv *env, jobject thisObj, jlong colAddr)
{
    return (*(SharedPtr<Collection<layers::backward::LayerIfacePtr > >*)colAddr)->size();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_BackwardLayers
 * Method:    cGet
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_BackwardLayers_cGet
(JNIEnv *env, jobject thisObj, jlong colAddr, jlong index)
{
    return (jlong) (new layers::backward::LayerIfacePtr
        ((*(SharedPtr<Collection<layers::backward::LayerIfacePtr > >*)colAddr)->get((size_t)index)));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_BackwardLayers
 * Method:    cPushBack
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_BackwardLayers_cPushBack
(JNIEnv *env, jobject thisObj, jlong colAddr, jlong layerAddr)
{
    (*(SharedPtr<Collection<layers::backward::LayerIfacePtr > >*)colAddr)->
        push_back(*((layers::backward::LayerIfacePtr *)layerAddr));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_BackwardLayers
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_BackwardLayers_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(SharedPtr<Collection<layers::backward::LayerIfacePtr > > *)addr;
}

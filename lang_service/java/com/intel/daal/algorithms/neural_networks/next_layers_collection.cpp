/* file: next_layers_collection.cpp */
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
#include "neural_networks/JNextLayersCollection.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_NextLayersCollection
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_NextLayersCollection_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new SharedPtr<Collection<layers::NextLayers> >(new Collection<layers::NextLayers>()));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_NextLayersCollection
 * Method:    cSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_NextLayersCollection_cSize
(JNIEnv *env, jobject thisObj, jlong colAddr)
{
    return (*(SharedPtr<Collection<layers::NextLayers > >*)colAddr)->size();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_NextLayersCollection
 * Method:    cGet
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_NextLayersCollection_cGet
(JNIEnv *env, jobject thisObj, jlong colAddr, jlong index)
{
    layers::NextLayers nextLayers = (*(SharedPtr<Collection<layers::NextLayers > >*)colAddr)->get((size_t)index);
    layers::NextLayers *nextLayersPtr = new layers::NextLayers(nextLayers);
    return (jlong)nextLayersPtr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_NextLayersCollection
 * Method:    cPushBack
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_NextLayersCollection_cPushBack
(JNIEnv *env, jobject thisObj, jlong colAddr, jlong layerAddr)
{
    (*(SharedPtr<Collection<layers::NextLayers > >*)colAddr)->push_back(*((layers::NextLayers *)layerAddr));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_NextLayersCollection
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_NextLayersCollection_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(SharedPtr<Collection<layers::NextLayers> > *)addr;
}

/* file: next_layers.cpp */
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
#include "neural_networks/layers/JNextLayers.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_NextLayers
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_NextLayers_cInit__
  (JNIEnv *env, jobject thisObj)
{
    return (jlong)(new layers::NextLayers());
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_NextLayers
 * Method:    cInit
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_NextLayers_cInit__J
  (JNIEnv *env, jobject thisObj, jlong index1)
{
    return (jlong)(new layers::NextLayers((size_t)index1));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_NextLayers
 * Method:    cInit
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_NextLayers_cInit__JJ
  (JNIEnv *env, jobject thisObj, jlong index1, jlong index2)
{
    return (jlong)(new layers::NextLayers((size_t)index1, (size_t)index2));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_NextLayers
 * Method:    cInit
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_NextLayers_cInit__JJJ
  (JNIEnv *env, jobject thisObj, jlong index1, jlong index2, jlong index3)
{
    return (jlong)(new layers::NextLayers((size_t)index1, (size_t)index2, (size_t)index3));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_NextLayers
 * Method:    cInit
 * Signature: (JJJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_NextLayers_cInit__JJJJ
  (JNIEnv *env, jobject thisObj, jlong index1, jlong index2, jlong index3, jlong index4)
{
    return (jlong)(new layers::NextLayers((size_t)index1, (size_t)index2, (size_t)index3, (size_t)index4));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_NextLayers
 * Method:    cInit
 * Signature: (JJJJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_NextLayers_cInit__JJJJJ
  (JNIEnv *env, jobject thisObj, jlong index1, jlong index2, jlong index3, jlong index4, jlong index5)
{
    return (jlong)(new layers::NextLayers((size_t)index1, (size_t)index2, (size_t)index3, (size_t)index4, (size_t)index5));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_NextLayers
 * Method:    cInit
 * Signature: (JJJJJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_NextLayers_cInit__JJJJJJ
  (JNIEnv *env, jobject thisObj, jlong index1, jlong index2, jlong index3, jlong index4, jlong index5, jlong index6)
{
    return (jlong)(new layers::NextLayers((size_t)index1, (size_t)index2, (size_t)index3, (size_t)index4, (size_t)index5, (size_t)index6));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_NextLayers
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_NextLayers_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(layers::NextLayers *)addr;
}

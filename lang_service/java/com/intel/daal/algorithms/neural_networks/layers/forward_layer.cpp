/* file: forward_layer.cpp */
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
#include "com_intel_daal_algorithms_neural_networks_layers_ForwardLayer.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ForwardLayer
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardLayer_cGetParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return (jlong) (*((forward::LayerIfacePtr *)algAddr))->getLayerParameter();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ForwardLayer
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardLayer_cGetInput
  (JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return (jlong) (*((forward::LayerIfacePtr *)algAddr))->getLayerInput();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ForwardLayer
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardLayer_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SerializationIfacePtr *ptr = new SerializationIfacePtr;
    *ptr = (*((forward::LayerIfacePtr *)algAddr))->getLayerResult();
    return (jlong)ptr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ForwardLayer
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardLayer_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(forward::LayerIfacePtr *)addr;
}

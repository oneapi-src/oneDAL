/* file: layer_descriptor.cpp */
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
#include "neural_networks/layers/JLayerDescriptor.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_LayerDescriptor
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_LayerDescriptor_cInit__
  (JNIEnv *env, jobject thisObj)
{
    return (jlong)(new layers::LayerDescriptor());
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_LayerDescriptor
 * Method:    cInit
 * Signature: (JJJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_LayerDescriptor_cInit__JJJ
  (JNIEnv *env, jobject thisObj, jlong index, jlong layerAddr, jlong nextLayersAddr)
{
    return (jlong)(new layers::LayerDescriptor((size_t)index,
                                               (*(layers::LayerIfacePtr*)layerAddr),
                                               (*(layers::NextLayers *)nextLayersAddr)));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_LayerDescriptor
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_LayerDescriptor_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(layers::LayerDescriptor *)addr;
}

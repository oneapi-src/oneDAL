/* file: layer_descriptor.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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

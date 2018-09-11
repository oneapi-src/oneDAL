/* file: backward_layer.cpp */
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
#include "neural_networks/layers/JBackwardLayer.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_BackwardLayer
 * Method:    cGetParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_BackwardLayer_cGetParameter
  (JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return (jlong) (*((backward::LayerIfacePtr *)algAddr))->getLayerParameter();
}


/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_BackwardLayer
 * Method:    cGetInput
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_BackwardLayer_cGetInput
  (JNIEnv *env, jobject thisObj, jlong algAddr)
{
    return (jlong) (*((backward::LayerIfacePtr *)algAddr))->getLayerInput();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_BackwardLayer
 * Method:    cGetResult
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_BackwardLayer_cGetResult
  (JNIEnv *env, jobject thisObj, jlong algAddr)
{
    SerializationIfacePtr *ptr = new SerializationIfacePtr;
    *ptr = (*((backward::LayerIfacePtr *)algAddr))->getLayerResult();
    return (jlong)ptr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_BackwardLayer
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_BackwardLayer_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(backward::LayerIfacePtr *)addr;
}

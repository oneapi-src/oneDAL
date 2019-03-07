/* file: forward_layers.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
#include "neural_networks/JForwardLayers.h"

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

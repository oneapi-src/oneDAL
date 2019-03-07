/* file: backward_layers.cpp */
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
#include "neural_networks/JBackwardLayers.h"

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

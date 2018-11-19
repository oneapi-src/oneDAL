/* file: prediction_topology.cpp */
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
#include "neural_networks/prediction/JPredictionTopology.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionTopology
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionTopology_cInit
  (JNIEnv *env, jobject thisObj)
{
    return (jlong)(new prediction::TopologyPtr(new prediction::Topology()));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionTopology
 * Method:    cSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionTopology_cSize
  (JNIEnv *env, jobject thisObj, jlong addr)
{
    return (*(prediction::TopologyPtr*)addr)->size();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionTopology
 * Method:    cGet
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionTopology_cGet
  (JNIEnv *env, jobject thisObj, jlong addr, jlong index)
{
    const layers::forward::LayerDescriptor& desc = (*(prediction::TopologyPtr*)addr)->get((size_t)index);
    layers::forward::LayerDescriptor *layerDescriptorPtr = new layers::forward::LayerDescriptor(desc);
    return (jlong)layerDescriptorPtr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionTopology
 * Method:    cPushBack
 * Signature: (JJ)V
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionTopology_cPushBack
(JNIEnv *env, jobject thisObj, jlong addr, jlong layerAddr)
{
    return (*(prediction::TopologyPtr*)addr)->add(*((layers::forward::LayerIfacePtr *)layerAddr));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionTopology
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionTopology_cDispose
  (JNIEnv *env, jobject thisObj, jlong addr)
{
    delete (prediction::TopologyPtr *)addr;
}

/*
* Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionTopology
* Method:    cAddNext
* Signature: (JJJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionTopology_cAddNext
(JNIEnv *, jobject, jlong addr, jlong index, jlong next)
{
    (*(prediction::TopologyPtr*)addr)->get(index).addNext(next);
}

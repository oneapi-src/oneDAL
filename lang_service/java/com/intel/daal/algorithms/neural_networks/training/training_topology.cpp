/* file: training_topology.cpp */
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
#include "neural_networks/training/JTrainingTopology.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;
/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingTopology
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingTopology_cInit
(JNIEnv *env, jobject thisObj)
{
    return (jlong)(new training::TopologyPtr(new training::Topology()));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingTopology
 * Method:    cSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingTopology_cSize
(JNIEnv *env, jobject thisObj, jlong addr)
{
    return (*(training::TopologyPtr*)addr)->size();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingTopology
 * Method:    cGet
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingTopology_cGet
(JNIEnv *env, jobject thisObj, jlong addr, jlong index)
{
    const layers::LayerDescriptor& desc = (*(training::TopologyPtr*)addr)->get((size_t)index);
    layers::LayerDescriptor *layerDescriptorPtr = new layers::LayerDescriptor(desc);
    return (jlong)layerDescriptorPtr;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingTopology
 * Method:    cPushBack
 * Signature: (JJ)V
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingTopology_cPushBack
(JNIEnv *env, jobject thisObj, jlong addr, jlong layerAddr)
{
    return (*(training::TopologyPtr*)addr)->add(*((layers::LayerIfacePtr *)layerAddr));
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_training_TrainingTopology
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingTopology_cDispose
(JNIEnv *env, jobject thisObj, jlong addr)
{
    delete(training::TopologyPtr *)addr;
}

/*
* Class:     com_intel_daal_algorithms_neural_networks_training_TrainingTopology
* Method:    cAddNext
* Signature: (JJJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_training_TrainingTopology_cAddNext
(JNIEnv *, jobject, jlong addr, jlong index, jlong next)
{
    (*(training::TopologyPtr*)addr)->get(index).addNext(next);
}

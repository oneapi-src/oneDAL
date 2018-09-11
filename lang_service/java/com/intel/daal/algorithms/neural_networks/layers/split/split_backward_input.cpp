/* file: split_backward_input.cpp */
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
#include "neural_networks/layers/split/JSplitBackwardInput.h"
#include "neural_networks/layers/split/JSplitBackwardInputLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define inputGradientCollectionId com_intel_daal_algorithms_neural_networks_layers_split_SplitBackwardInputLayerDataId_inputGradientCollectionId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers::split;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitBackwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitBackwardInput_cSetInput__JIJ
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == inputGradientCollectionId)
    {
        jniInput<backward::Input>::set<backward::InputLayerDataId, KeyValueDataCollection>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitBackwardInput
 * Method:    cSetInput
 * Signature: (JIJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitBackwardInput_cSetInput__JIJJ
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr, jlong index)
{
    if (id == inputGradientCollectionId)
    {
        jniInput<backward::Input>::set<backward::InputLayerDataId, Tensor>(inputAddr, id, ntAddr, (size_t)index);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitBackwardInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitBackwardInput_cGetInput__JI
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == inputGradientCollectionId)
    {
        return jniInput<backward::Input>::get<backward::InputLayerDataId, KeyValueDataCollection>(inputAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_split_SplitBackwardInput
 * Method:    cGetInput
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_split_SplitBackwardInput_cGetInput__JIJ
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong index)
{
    if (id == inputGradientCollectionId)
    {
        return jniInput<backward::Input>::get<backward::InputLayerDataId, Tensor>(inputAddr, id, (size_t)index);
    }

    return (jlong)0;
}

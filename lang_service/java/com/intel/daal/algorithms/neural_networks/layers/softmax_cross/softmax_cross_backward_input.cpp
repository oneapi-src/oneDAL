/* file: softmax_cross_backward_input.cpp */
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
#include "neural_networks/layers/softmax_cross/JSoftmaxCrossBackwardInput.h"
#include "neural_networks/layers/softmax_cross/JSoftmaxCrossLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define auxProbabilitiesId com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossLayerDataId_auxProbabilitiesId
#define auxGroundTruthId com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossLayerDataId_auxGroundTruthId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers::loss::softmax_cross;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossBackwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossBackwardInput_cSetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == auxGroundTruthId || id == auxProbabilitiesId)
    {
        jniInput<backward::Input>::set<LayerDataId, Tensor>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_softmax_cross_SoftmaxCrossBackwardInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_softmax_1cross_SoftmaxCrossBackwardInput_cGetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == auxGroundTruthId || id == auxProbabilitiesId)
    {
        return jniInput<backward::Input>::get<LayerDataId, Tensor>(inputAddr, id);
    }

    return (jlong)0;
}

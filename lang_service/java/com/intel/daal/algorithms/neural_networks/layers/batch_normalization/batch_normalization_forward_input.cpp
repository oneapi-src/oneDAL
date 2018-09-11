/* file: batch_normalization_forward_input.cpp */
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
#include "neural_networks/layers/batch_normalization/JBatchNormalizationForwardInput.h"
#include "neural_networks/layers/batch_normalization/JBatchNormalizationForwardInputLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define populationMeanId com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationForwardInputLayerDataId_populationMeanId
#define populationVarianceId com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationForwardInputLayerDataId_populationVarianceId

USING_COMMON_NAMESPACES()

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationForwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationForwardInput_cSetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    using namespace daal::algorithms::neural_networks::layers::batch_normalization;
    if (id == populationMeanId || id == populationVarianceId)
    {
        jniInput<forward::Input>::set<forward::InputLayerDataId, Tensor>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationForwardInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationForwardInput_cGetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    using namespace daal::algorithms::neural_networks::layers::batch_normalization;
    if (id == populationMeanId || id == populationVarianceId)
    {
        return jniInput<forward::Input>::get<forward::InputLayerDataId, Tensor>(inputAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationForwardInput
 * Method:    cGetWeightsSizes
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationForwardInput_cGetWeightsSizes
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jlong paramAddr)
{
    using namespace daal::algorithms::neural_networks::layers;
    Collection<size_t> dims = ((batch_normalization::forward::Input *)inputAddr)->getWeightsSizes((batch_normalization::Parameter *)paramAddr);
    return getJavaLongArrayFromSizeTCollection(env, dims);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_batch_normalization_BatchNormalizationForwardInput
 * Method:    cGetBiasesSizes
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_batch_1normalization_BatchNormalizationForwardInput_cGetBiasesSizes
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jlong paramAddr)
{
    using namespace daal::algorithms::neural_networks::layers;
    Collection<size_t> dims = ((batch_normalization::forward::Input *)inputAddr)->getBiasesSizes((batch_normalization::Parameter *)paramAddr);
    return getJavaLongArrayFromSizeTCollection(env, dims);
}

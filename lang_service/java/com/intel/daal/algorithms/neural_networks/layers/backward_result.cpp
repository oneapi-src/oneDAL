/* file: backward_result.cpp */
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
#include "neural_networks/layers/JBackwardResult.h"
#include "neural_networks/layers/JBackwardResultId.h"
#include "neural_networks/layers/JBackwardResultLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define gradientId com_intel_daal_algorithms_neural_networks_layers_BackwardResultId_gradientId
#define weightDerivativesId com_intel_daal_algorithms_neural_networks_layers_BackwardResultId_weightDerivativesId
#define biasDerivativesId com_intel_daal_algorithms_neural_networks_layers_BackwardResultId_biasDerivativesId
#define resultLayerDataId com_intel_daal_algorithms_neural_networks_layers_BackwardResultLayerDataId_resultLayerDataId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_BackwardResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_BackwardResult_cGetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == gradientId || id == weightDerivativesId || id == biasDerivativesId)
    {
        return jniArgument<backward::Result>::get<backward::ResultId, Tensor>(resAddr, id);
    } else if (id == resultLayerDataId)
    {
        return jniArgument<backward::Result>::get<backward::ResultLayerDataId, KeyValueDataCollection>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_BackwardResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_BackwardResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    if (id == gradientId || id == weightDerivativesId || id == biasDerivativesId)
    {
        jniArgument<backward::Result>::set<backward::ResultId, Tensor>(resAddr, id, ntAddr);
    } else if (id == resultLayerDataId)
    {
        jniArgument<backward::Result>::set<backward::ResultLayerDataId, KeyValueDataCollection>(resAddr, id, ntAddr);
    }
}

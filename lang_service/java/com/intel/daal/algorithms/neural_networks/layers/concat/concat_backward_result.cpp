/* file: concat_backward_result.cpp */
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
#include "neural_networks/layers/JBackwardResultLayerDataId.h"
#include "neural_networks/layers/concat/JConcatBackwardResult.h"

#include "daal.h"

#include "common_helpers.h"

#define resultLayerDataId com_intel_daal_algorithms_neural_networks_layers_BackwardResultLayerDataId_resultLayerDataId

using namespace daal;
using namespace daal::algorithms::neural_networks::layers::concat;
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_concat_ConcatBackwardResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_concat_ConcatBackwardResult_cNewResult
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<backward::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ConcatBackwardResult
 * Method:    cGetValue
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_concat_ConcatBackwardResult_cGetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong index)
{
    if (id == resultLayerDataId)
    {
        return jniArgument<backward::Result>::get<layers::backward::ResultLayerDataId, Tensor>(resAddr, id, (size_t)index);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ConcatBackwardResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_concat_ConcatBackwardResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr, jlong index)
{
    if (id == resultLayerDataId)
    {
        jniArgument<backward::Result>::set<layers::backward::ResultLayerDataId, Tensor>(resAddr, id, ntAddr, (size_t)index);
    }
}

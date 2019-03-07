/* file: forward_result.cpp */
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
#include "neural_networks/layers/JForwardResult.h"
#include "neural_networks/layers/JForwardResultId.h"
#include "neural_networks/layers/JForwardResultLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define valueId com_intel_daal_algorithms_neural_networks_layers_ForwardResultId_valueId
#define resultForBackwardId com_intel_daal_algorithms_neural_networks_layers_ForwardResultLayerDataId_resultForBackwardId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ForwardResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardResult_cGetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == valueId)
    {
        return jniArgument<forward::Result>::get<forward::ResultId, Tensor>(resAddr, id);
    } else if (id == resultForBackwardId)
    {
        return jniArgument<forward::Result>::get<forward::ResultLayerDataId, KeyValueDataCollection>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ForwardResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    if (id == valueId)
    {
        jniArgument<forward::Result>::set<forward::ResultId, Tensor>(resAddr, id, ntAddr);
    } else if (id == resultForBackwardId)
    {
        jniArgument<forward::Result>::set<forward::ResultLayerDataId, KeyValueDataCollection>(resAddr, id, ntAddr);
    }
}

/* file: reshape_forward_result.cpp */
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
#include "neural_networks/layers/reshape/JReshapeForwardResult.h"
#include "neural_networks/layers/reshape/JReshapeLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define auxDataId com_intel_daal_algorithms_neural_networks_layers_reshape_ReshapeLayerDataId_auxDataId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers::reshape;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_reshape_ReshapeForwardResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_reshape_ReshapeForwardResult_cNewResult
  (JNIEnv *env, jobject thisObj)
{
    return jniArgument<forward::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_reshape_ReshapeForwardResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_reshape_ReshapeForwardResult_cGetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == auxDataId)
    {
        return jniArgument<forward::Result>::get<LayerDataId, NumericTable>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_reshape_ReshapeForwardResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_reshape_ReshapeForwardResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    if (id == auxDataId)
    {
        jniArgument<forward::Result>::set<LayerDataId, NumericTable>(resAddr, id, ntAddr);
    }
}

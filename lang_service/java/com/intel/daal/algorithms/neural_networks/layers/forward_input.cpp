/* file: forward_input.cpp */
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
#include "neural_networks/layers/JForwardInput.h"
#include "neural_networks/layers/JForwardInputId.h"
#include "neural_networks/layers/JForwardInputLayerDataId.h"

#include "daal.h"

#include "common_helpers.h"

#define dataId com_intel_daal_algorithms_neural_networks_layers_ForwardInputId_dataId
#define weightsId com_intel_daal_algorithms_neural_networks_layers_ForwardInputId_weightsId
#define biasesId com_intel_daal_algorithms_neural_networks_layers_ForwardInputId_biasesId
#define inputLayerDataId com_intel_daal_algorithms_neural_networks_layers_ForwardInputLayerDataId_inputLayerDataId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::neural_networks::layers;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ForwardInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardInput_cSetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == dataId || id == weightsId || id == biasesId)
    {
        jniInput<forward::Input>::set<forward::InputId, Tensor>(inputAddr, id, ntAddr);
    } else if (id == inputLayerDataId)
    {
        jniInput<forward::Input>::set<forward::InputLayerDataId, KeyValueDataCollection>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_layers_ForwardInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_layers_ForwardInput_cGetInput
  (JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == dataId || id == weightsId || id == biasesId)
    {
        return jniInput<forward::Input>::get<forward::InputId, Tensor>(inputAddr, id);
    } else if (id == inputLayerDataId)
    {
        return jniInput<forward::Input>::get<forward::InputLayerDataId, KeyValueDataCollection>(inputAddr, id);
    }

    return (jlong)0;
}

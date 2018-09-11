/* file: prediction_input.cpp */
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
#include "neural_networks/prediction/JPredictionInput.h"
#include "neural_networks/prediction/JPredictionTensorInputId.h"
#include "neural_networks/prediction/JPredictionModelInputId.h"

#include "daal.h"

#include "common_helpers.h"

#define dataId com_intel_daal_algorithms_neural_networks_prediction_PredictionTensorInputId_dataId
#define modelId com_intel_daal_algorithms_neural_networks_prediction_PredictionModelInputId_modelId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if (id == dataId)
    {
        jniInput<prediction::Input>::set<prediction::TensorInputId, Tensor>(inputAddr, id, ntAddr);
    } else if (id == modelId)
    {
        jniInput<prediction::Input>::set<prediction::ModelInputId, prediction::Model>(inputAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == dataId)
    {
        return jniInput<prediction::Input>::get<prediction::TensorInputId, Tensor>(inputAddr, id);
    } else if (id == modelId)
    {
        return jniInput<prediction::Input>::get<prediction::ModelInputId, prediction::Model>(inputAddr, id);
    }

    return (jlong)0;
}

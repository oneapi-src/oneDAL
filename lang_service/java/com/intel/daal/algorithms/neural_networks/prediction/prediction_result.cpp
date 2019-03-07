/* file: prediction_result.cpp */
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
#include "neural_networks/prediction/JPredictionResult.h"
#include "neural_networks/prediction/JPredictionResultId.h"
#include "neural_networks/prediction/JPredictionResultCollectionId.h"

#include "daal.h"

#include "common_helpers.h"

#define predictionId com_intel_daal_algorithms_neural_networks_prediction_PredictionResultId_predictionId
#define predictionCollectionId com_intel_daal_algorithms_neural_networks_prediction_PredictionResultCollectionId_predictionCollectionId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionResult_cGetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == predictionId)
    {
        return jniArgument<prediction::Result>::get<prediction::ResultId, Tensor>(resAddr, id);
    }
    else if (id == predictionCollectionId)
    {
        return jniArgument<prediction::Result>::get<prediction::ResultCollectionId, KeyValueDataCollection>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    if (id == predictionId)
    {
        jniArgument<prediction::Result>::set<prediction::ResultId, Tensor>(resAddr, id, ntAddr);
    }
    else if (id == predictionCollectionId)
    {
        jniArgument<prediction::Result>::set<prediction::ResultCollectionId, KeyValueDataCollection>(resAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionResult
 * Method:    cAddTensor
 * Signature: (JIIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionResult_cAddTensor
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jint key, jlong ntAddr)
{
    if (id == predictionCollectionId)
    {
        SerializationIfacePtr *collectionShPtr = (SerializationIfacePtr *)(jniArgument<prediction::Result>::get<prediction::ResultCollectionId, KeyValueDataCollection>(resAddr, id));
        SerializationIfacePtr *valueShPtr = (SerializationIfacePtr *)ntAddr;
        KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
        (*collection)[(size_t)key] = *valueShPtr;
    }
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionResult
 * Method:    cGetTensor
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionResult_cGetTensor
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jint key)
{
    if (id == predictionCollectionId)
    {
        SerializationIfacePtr *collectionShPtr = (SerializationIfacePtr *)(jniArgument<prediction::Result>::get<prediction::ResultCollectionId, KeyValueDataCollection>(resAddr, id));
        KeyValueDataCollection *collection = static_cast<KeyValueDataCollection *>(collectionShPtr->get());
        SerializationIfacePtr *value = new SerializationIfacePtr((*collection)[(size_t)key]);
        return (jlong)value;
    }

    return (jlong)0;
}

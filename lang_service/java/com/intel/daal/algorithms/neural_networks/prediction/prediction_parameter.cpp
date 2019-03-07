/* file: prediction_parameter.cpp */
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
#include "neural_networks/prediction/JPredictionParameter.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::neural_networks;

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionParameter
 * Method:    cInit
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionParameter_cInit
  (JNIEnv *env, jobject thisObj)
{
    return (jlong)(new prediction::Parameter());
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionParameter
 * Method:    cGetBatchSize
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionParameter_cGetBatchSize
  (JNIEnv *env, jobject thisObj, jlong addr)
{
    return (jlong)(((prediction::Parameter *)addr)->batchSize);
}

/*
 * Class:     com_intel_daal_algorithms_neural_networks_prediction_PredictionParameter
 * Method:    cSetBatchSize
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_neural_1networks_prediction_PredictionParameter_cSetBatchSize
  (JNIEnv *env, jobject thisObj, jlong addr, jlong batchSize)
{
    ((prediction::Parameter *)addr)->batchSize = batchSize;
}

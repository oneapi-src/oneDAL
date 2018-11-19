/* file: prediction_result.cpp */
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
#include "logistic_regression/prediction/JPredictionResult.h"
#include "logistic_regression/prediction/JPredictionResultNumericTableId.h"
#include "classifier/prediction/JPredictionResultId.h"
#include "logistic_regression/prediction/JPredictionMethod.h"

#include "daal.h"

#include "common_helpers.h"

#define predictionResultId com_intel_daal_algorithms_classifier_prediction_PredictionResultId_Prediction
#define probabilitiesId com_intel_daal_algorithms_logistic_regression_prediction_PredictionResultNumericTableId_probabilitiesValue
#define logProbabilitiesId com_intel_daal_algorithms_logistic_regression_prediction_PredictionResultNumericTableId_logProbabilitiesValue

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionResult
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionResult_cGetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == predictionResultId)
    {
        return jniArgument<logistic_regression::prediction::Result>::get<classifier::prediction::ResultId, NumericTable>(resAddr, id);
    }
    else if (id == probabilitiesId || id == logProbabilitiesId)
    {
        return jniArgument<logistic_regression::prediction::Result>::get<logistic_regression::prediction::ResultNumericTableId, NumericTable>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionResult
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionResult_cSetValue
  (JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    if (id == predictionResultId)
    {
        jniArgument<logistic_regression::prediction::Result>::set<classifier::prediction::ResultId, NumericTable>(resAddr, id, ntAddr);
    }
    else if (id == probabilitiesId || id == logProbabilitiesId)
    {
        jniArgument<logistic_regression::prediction::Result>::set<logistic_regression::prediction::ResultNumericTableId, NumericTable>(resAddr, id, ntAddr);
    }
}

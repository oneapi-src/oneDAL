/* file: predict_result.cpp */
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

#include "linear_regression/prediction/JPredictionResult.h"
#include "linear_regression/prediction/JPredictionResultId.h"
#include "linear_regression/prediction/JPredictionMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::linear_regression::prediction;

/*
 * Class:     com_intel_daal_algorithms_linear_regression_prediction_PredictionResult
 * Method:    cNewResult
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_prediction_PredictionResult_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<linear_regression::prediction::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_prediction_PredictionResult
 * Method:    cGetPredictionResult
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_prediction_PredictionResult_cGetPredictionResult
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<linear_regression::prediction::Result>::get<linear_regression::prediction::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_prediction_PredictionResult
 * Method:    cSetPredictionResult
 * Signature:(JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_linear_1regression_prediction_PredictionResult_cSetPredictionResult
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<linear_regression::prediction::Result>::set<linear_regression::prediction::ResultId, NumericTable>(resAddr, id, ntAddr);
}

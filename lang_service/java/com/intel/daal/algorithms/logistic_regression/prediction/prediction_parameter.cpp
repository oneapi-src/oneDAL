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

#include "daal.h"
#include "logistic_regression/prediction/JPredictionParameter.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionParameter
 * Method:    cParInit
 * Signature:()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionParameter_cParInit
(JNIEnv *env, jobject thisObj)
{
    return(jlong)new logistic_regression::prediction::Parameter();
}

/*
 * Class:     com_intel_daal_algorithms_logistic_1regression_PredictionParameter
 * Method:    cSetSeed
 * Signature:(JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionParameter_cSetResultsToCompute
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong resToCompute)
{
    (*(logistic_regression::prediction::Parameter *)parAddr).resultsToCompute = resToCompute;
}

/*
 * Class:     com_intel_daal_algorithms_logistic_1regression_PredictionParameter
 * Method:    cGetSeed
 * Signature:(J)I
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionParameter_cGetResultsToCompute
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(jlong)(*(logistic_regression::prediction::Parameter *)parAddr).resultsToCompute;
}

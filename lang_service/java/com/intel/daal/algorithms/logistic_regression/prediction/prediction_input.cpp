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

#include "logistic_regression/prediction/JPredictionInput.h"
#include "logistic_regression/prediction/JPredictionInputId.h"
#include "logistic_regression/prediction/JPredictionMethod.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::logistic_regression::prediction;

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionInput
 * Method:    cSetInputTable
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionInput_cSetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id != classifier::prediction::data) return;

    jniInput<logistic_regression::prediction::Input>::set<classifier::prediction::NumericTableInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionInput
 * Method:    cSetInputModel
 * Signature:(JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionInput_cSetInputModel
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id != classifier::prediction::model) return;

    jniInput<logistic_regression::prediction::Input>::set<classifier::prediction::ModelInputId, logistic_regression::Model>(inputAddr, id, ntAddr);
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionInput_cGetInputTable
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if(id != classifier::prediction::data) return (jlong)-1;

    return jniInput<logistic_regression::prediction::Input>::get<classifier::prediction::NumericTableInputId, NumericTable>(inputAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_logistic_regression_prediction_PredictionInput
 * Method:    cGetInputTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logistic_1regression_prediction_PredictionInput_cGetInputModel
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if(id != classifier::prediction::model) return (jlong)-1;

    return jniInput<logistic_regression::prediction::Input>::get<classifier::prediction::ModelInputId, logistic_regression::Model>(inputAddr, id);
}

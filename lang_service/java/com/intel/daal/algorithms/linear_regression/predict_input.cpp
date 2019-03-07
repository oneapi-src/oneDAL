/* file: predict_input.cpp */
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

#include "linear_regression/prediction/JInput.h"
#include "linear_regression/prediction/JPredictionInputId.h"
#include "linear_regression/prediction/JPredictionMethod.h"

#include "common_helpers.h"

#define defaultDenseValue com_intel_daal_algorithms_linear_regression_prediction_PredictionMethod_defaultDenseValue

#define dataId com_intel_daal_algorithms_linear_regression_prediction_PredictionInputId_dataId
#define modelId com_intel_daal_algorithms_linear_regression_prediction_PredictionInputId_modelId

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::linear_regression::prediction;

/*
 * Class:     com_intel_daal_algorithms_linear_regression_prediction_Input
 * Method:    cSetInput
 * Signature:(JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_linear_1regression_prediction_Input_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong ntAddr)
{
    if(id == dataId)
    {
        jniInput<linear_regression::prediction::Input>::
            set<linear_regression::prediction::NumericTableInputId, NumericTable>(inputAddr, linear_regression::prediction::data, ntAddr);
    }
    else if(id == modelId)
    {
        jniInput<linear_regression::prediction::Input>::
            set<linear_regression::prediction::ModelInputId, linear_regression::Model>(inputAddr, linear_regression::prediction::model, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_prediction_PredictionBatch
 * Method:    cGetInput
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_prediction_Input_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if(id == dataId)
    {
        return jniInput<linear_regression::prediction::Input>::
            get<linear_regression::prediction::NumericTableInputId, NumericTable>(inputAddr, linear_regression::prediction::data);
    }
    else if(id == modelId)
    {
        return jniInput<linear_regression::prediction::Input>::
            get<linear_regression::prediction::ModelInputId, linear_regression::Model>(inputAddr, linear_regression::prediction::model);
    }

    return (jlong)0;
}

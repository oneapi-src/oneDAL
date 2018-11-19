/* file: train_result.cpp */
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

#include "JComputeMode.h"
#include "JComputeStep.h"
#include "linear_regression/training/JTrainingResult.h"
#include "linear_regression/training/JTrainingResultId.h"
#include "linear_regression/training/JTrainingMethod.h"

#include "common_helpers.h"

#define ModelId     com_intel_daal_algorithms_linear_regression_training_TrainingResultId_ModelId

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingResult
 * Method:    cNewResult
 * Signature: ()J
 */

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingResult_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<linear_regression::training::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingResult
 * Method:    cGetModel
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingResult_cGetModel
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if ( id == ModelId )
    {
        return jniArgument<linear_regression::training::Result>::
            get<linear_regression::training::ResultId, linear_regression::Model>(resAddr, id);
    }

    return (jlong)0;
}

/* file: train_result.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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

/* file: training_result.cpp */
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

#include "daal.h"

#include "implicit_als/training/JTrainingResult.h"

#include "implicit_als_training_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_TrainingResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_TrainingResult_cNewResult
(JNIEnv *, jobject)
{
    return jniArgument<implicit_als::training::Result>::newObj();
}
/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_TrainingResult
 * Method:    cGetResult
 * Signature: (JIIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_TrainingResult_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<implicit_als::training::Method, Batch, fastCSR, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_TrainingResult
 * Method:    cGetResultModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_TrainingResult_cGetResultModel
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if(id == modelId)
    {
        return jniArgument<implicit_als::training::Result>::get<ResultId, implicit_als::Model>(resAddr, model);
    }
    else
    {
        return (jlong)0;
    }
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_TrainingResult
 * Method:    cSetResultModel
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_TrainingResult_cSetResultModel
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong mdlAddr)
{
    if(id == modelId)
    {
        jniArgument<implicit_als::training::Result>::set<ResultId, implicit_als::Model>(resAddr, model, mdlAddr);
    }
}

/* file: training_init_result.cpp */
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

#include "implicit_als/training/init/JInitResult.h"

#include "implicit_als_init_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als;
using namespace daal::algorithms::implicit_als::training::init;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitResult_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<implicit_als::training::init::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitResult
 * Method:    cGetResult
 * Signature: (JIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitResult_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method, jint cmode)
{
    if (cmode == jBatch)
    {
        return jniBatch<implicit_als::training::init::Method, Batch, fastCSR, defaultDense>::getResult(prec, method, algAddr);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitResult
 * Method:    cGetResultModel
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitResult_cGetResultModel
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if (id == initModelId)
    {
        return jniArgument<implicit_als::training::init::Result>::
            get<training::init::ResultId, implicit_als::Model>(resAddr, training::init::ResultId::model);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitResult
 * Method:    cSetResultModel
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitResult_cSetResultModel
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong mdlAddr)
{
    if (id == initModelId)
    {
        jniArgument<implicit_als::training::init::Result>::
            set<training::init::ResultId, implicit_als::Model>(resAddr, training::init::ResultId::model, mdlAddr);
    }
}

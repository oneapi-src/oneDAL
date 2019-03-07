/* file: training_init_result.cpp */
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

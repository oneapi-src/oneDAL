/* file: training_result.cpp */
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

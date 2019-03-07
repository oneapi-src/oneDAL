/* file: training_input.cpp */
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

#include "implicit_als/training/JTrainingInput.h"

#include "implicit_als_training_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_TrainingInput
 * Method:    cInit
 * Signature: (JIIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_TrainingInput_cInit
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<implicit_als::training::Method, Batch, fastCSR, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_TrainingInput
 * Method:    cSetInput
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_TrainingInput_cSetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong valAddr)
{
    if (id == dataId)
    {
        jniInput<implicit_als::training::Input>::set<NumericTableInputId, NumericTable>(inputAddr, data, valAddr);
    }
    else if (id == inputModelId)
    {
        jniInput<implicit_als::training::Input>::set<ModelInputId, implicit_als::Model>(inputAddr, inputModel, valAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_TrainingInput
 * Method:    cGetInput
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_TrainingInput_cGetInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id)
{
    if (id == dataId)
    {
        return jniInput<implicit_als::training::Input>::get<NumericTableInputId, NumericTable>(inputAddr, data);
    }
    else if (id == inputModelId)
    {
        return jniInput<implicit_als::training::Input>::get<ModelInputId, implicit_als::Model>(inputAddr, inputModel);
    }

    return (jlong)0;
}

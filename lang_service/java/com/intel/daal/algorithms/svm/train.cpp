/* file: train.cpp */
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
#include "JComputeMode.h"
#include "classifier/training/JInputId.h"
#include "svm/training/JTrainingBatch.h"
#include "classifier/training/JTrainingInput.h"
#include "svm/training/JTrainingMethod.h"
#include "svm/training/JTrainingResult.h"
#include "classifier/training/JTrainingResultId.h"
#include "daal.h"

#include "common_helpers.h"

const int innerModelId        = com_intel_daal_algorithms_classifier_training_TrainingResultId_Model;

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::svm::training;

/*
 * Class:     com_intel_daal_algorithms_svm_training_TrainingBatch
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_training_TrainingBatch_cInit
(JNIEnv *env, jobject obj, jint prec, jint method)
{
    return jniBatch<svm::training::Method, svm::training::Batch, boser>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_svm_training_TrainingBatch
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_training_TrainingBatch_cInitParameter
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<svm::training::Method, svm::training::Batch, boser>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svm_training_TrainingBatch
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_training_TrainingBatch_cGetInput
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<svm::training::Method, svm::training::Batch, boser>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svm_training_TrainingBatch
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_training_TrainingBatch_cGetResult
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<svm::training::Method, svm::training::Batch, boser>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_svm_training_TrainingResult
 * Method:    cGetModel
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_training_TrainingResult_cGetModel
(JNIEnv *env, jobject obj, jlong resAddr, jint id)
{
    if (id == innerModelId)
    {
        return jniArgument<svm::training::Result>::get<classifier::training::ResultId, svm::Model>(resAddr, classifier::training::model);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_svm_training_TrainingBatch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_svm_training_TrainingBatch_cClone
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<svm::training::Method, svm::training::Batch, boser>::getClone(prec, method, algAddr);
}

/* file: train_batch.cpp */
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

#include "linear_regression/training/JTrainingBatch.h"
#include "linear_regression/training/JTrainingMethod.h"

#include "common_helpers.h"

#define normEqDense com_intel_daal_algorithms_linear_regression_training_TrainingMethod_normEqDenseValue
#define qrDense     com_intel_daal_algorithms_linear_regression_training_TrainingMethod_qrDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::linear_regression::training;

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_BatchTraining
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<linear_regression::training::Method, Batch, normEqDense, qrDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_BatchTraining
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingBatch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<linear_regression::training::Method, Batch, normEqDense, qrDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_BatchTraining
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingBatch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<linear_regression::training::Method, Batch, normEqDense, qrDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_BatchTraining
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingBatch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<linear_regression::training::Method, Batch, normEqDense, qrDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_BatchTraining
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingBatch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<linear_regression::training::Method, Batch, normEqDense, qrDense>::getClone(prec, method, algAddr);
}

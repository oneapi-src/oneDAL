/* file: train_batch.cpp */
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

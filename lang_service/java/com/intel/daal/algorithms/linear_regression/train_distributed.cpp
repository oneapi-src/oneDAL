/* file: train_distributed.cpp */
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

#include "JComputeStep.h"
#include "linear_regression/training/JInput.h"
#include "linear_regression/training/JTrainingDistributedStep1Local.h"
#include "linear_regression/training/JTrainingDistributedStep2Master.h"
#include "linear_regression/training/JTrainingMethod.h"

#include "common_helpers.h"

#define normEqDense com_intel_daal_algorithms_linear_regression_training_TrainingMethod_normEqDenseValue
#define qrDense     com_intel_daal_algorithms_linear_regression_training_TrainingMethod_qrDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::linear_regression::training;

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep1Local
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep1Local_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniDistributed<step1Local, linear_regression::training::Method, Distributed, normEqDense, qrDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep1Local
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep1Local_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, linear_regression::training::Method, Distributed, normEqDense, qrDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep1Local
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep1Local_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, linear_regression::training::Method, Distributed, normEqDense, qrDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep1Local
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep1Local_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, linear_regression::training::Method, Distributed, normEqDense, qrDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep1Local
 * Method:    cGetPartialResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep1Local_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, linear_regression::training::Method, Distributed, normEqDense, qrDense>::
        getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep1Local
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep1Local_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step1Local, linear_regression::training::Method, Distributed, normEqDense, qrDense>::getClone(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep2Master
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep2Master_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniDistributed<step2Master, linear_regression::training::Method, Distributed, normEqDense, qrDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep2Master
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep2Master_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, linear_regression::training::Method, Distributed, normEqDense, qrDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep2Master
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep2Master_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, linear_regression::training::Method, Distributed, normEqDense, qrDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep2Master
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep2Master_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, linear_regression::training::Method, Distributed, normEqDense, qrDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep2Master
 * Method:    cGetPartialResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep2Master_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, linear_regression::training::Method, Distributed, normEqDense, qrDense>::
        getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingDistributedStep2Master
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingDistributedStep2Master_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, linear_regression::training::Method, Distributed, normEqDense, qrDense>::getClone(prec, method, algAddr);
}

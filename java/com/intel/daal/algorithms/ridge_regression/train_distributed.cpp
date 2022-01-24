/* file: train_distributed.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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

#include "com_intel_daal_algorithms_ridge_regression_training_Input.h"
#include "com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master.h"

#include "com/intel/daal/common_helpers.h"

#include "com_intel_daal_algorithms_ridge_regression_training_TrainingMethod.h"
#define normEqDense com_intel_daal_algorithms_ridge_regression_training_TrainingMethod_normEqDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::ridge_regression::training;

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cInit(JNIEnv * env, jobject thisObj,
                                                                                                                       jint prec, jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cInitTrainParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cInitTrainParameter(
    JNIEnv * env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cGetInput(JNIEnv * env,
                                                                                                                           jobject thisObj,
                                                                                                                           jlong algAddr, jint prec,
                                                                                                                           jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cGetResult(JNIEnv * env,
                                                                                                                            jobject thisObj,
                                                                                                                            jlong algAddr, jint prec,
                                                                                                                            jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cGetPartialResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cGetPartialResult(
    JNIEnv * env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cClone(JNIEnv * env, jobject thisObj,
                                                                                                                        jlong algAddr, jint prec,
                                                                                                                        jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::getClone(prec, method, algAddr);
}

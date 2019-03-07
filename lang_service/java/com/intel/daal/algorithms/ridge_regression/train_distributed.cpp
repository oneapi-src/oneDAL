/* file: train_distributed.cpp */
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

#include "JComputeStep.h"
#include "ridge_regression/training/JInput.h"
#include "ridge_regression/training/JTrainingDistributedStep2Master.h"
#include "ridge_regression/training/JTrainingMethod.h"

#include "common_helpers.h"

#define normEqDense com_intel_daal_algorithms_ridge_regression_training_TrainingMethod_normEqDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::ridge_regression::training;

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cInitTrainParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cInitTrainParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cGetPartialResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingDistributedStep2Master
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingDistributedStep2Master_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniDistributed<step2Master, ridge_regression::training::Method, Distributed, normEqDense>::getClone(prec, method, algAddr);
}

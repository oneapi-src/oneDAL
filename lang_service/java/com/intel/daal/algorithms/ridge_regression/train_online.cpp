/* file: train_online.cpp */
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
#include "ridge_regression/training/JTrainingOnline.h"
#include "ridge_regression/training/JTrainingMethod.h"

#include "common_helpers.h"

#define normEqDense com_intel_daal_algorithms_ridge_regression_training_TrainingMethod_normEqDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::ridge_regression::training;

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingOnline
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingOnline_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniOnline<ridge_regression::training::Method, Online, normEqDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingOnline
 * Method:    cInitTrainParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingOnline_cInitTrainParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<ridge_regression::training::Method, Online, normEqDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingOnline
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingOnline_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<ridge_regression::training::Method, Online, normEqDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingOnline
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingOnline_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<ridge_regression::training::Method, Online, normEqDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingOnline
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingOnline_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<ridge_regression::training::Method, Online, normEqDense>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_ridge_regression_training_TrainingOnline
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_ridge_1regression_training_TrainingOnline_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<ridge_regression::training::Method, Online, normEqDense>::getClone(prec, method, algAddr);
}

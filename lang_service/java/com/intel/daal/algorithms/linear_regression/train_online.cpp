/* file: train_online.cpp */
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
#include "linear_regression/training/JTrainingOnline.h"
#include "linear_regression/training/JTrainingMethod.h"

#include "common_helpers.h"

#define normEqDense com_intel_daal_algorithms_linear_regression_training_TrainingMethod_normEqDenseValue
#define qrDense     com_intel_daal_algorithms_linear_regression_training_TrainingMethod_qrDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::linear_regression::training;

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingOnline
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingOnline_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniOnline<linear_regression::training::Method, Online, normEqDense, qrDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingOnline
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingOnline_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<linear_regression::training::Method, Online, normEqDense, qrDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingOnline
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingOnline_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<linear_regression::training::Method, Online, normEqDense, qrDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingOnline
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingOnline_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<linear_regression::training::Method, Online, normEqDense, qrDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingOnline
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingOnline_cGetPartialResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<linear_regression::training::Method, Online, normEqDense, qrDense>::getPartialResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_linear_regression_training_TrainingOnline
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_linear_1regression_training_TrainingOnline_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniOnline<linear_regression::training::Method, Online, normEqDense, qrDense>::getClone(prec, method, algAddr);
}

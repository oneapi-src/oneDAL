/* file: train_batch.cpp */
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

#include "kdtree_knn_classification/training/JTrainingBatch.h"
#include "kdtree_knn_classification/training/JTrainingMethod.h"

#include "common_helpers.h"

#define defaultDense com_intel_daal_algorithms_kdtree_knn_classification_training_TrainingMethod_defaultDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::kdtree_knn_classification::training;

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_training_BatchTraining
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_training_TrainingBatch_cInit
(JNIEnv *env, jobject thisObj, jint prec, jint method)
{
    return jniBatch<kdtree_knn_classification::training::Method, Batch, defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_training_TrainingBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_training_TrainingBatch_cInitParameter
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kdtree_knn_classification::training::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_training_BatchTraining
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_training_TrainingBatch_cGetInput
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kdtree_knn_classification::training::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_training_BatchTraining
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_training_TrainingBatch_cGetResult
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kdtree_knn_classification::training::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_training_BatchTraining
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_training_TrainingBatch_cClone
(JNIEnv *env, jobject thisObj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<kdtree_knn_classification::training::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}

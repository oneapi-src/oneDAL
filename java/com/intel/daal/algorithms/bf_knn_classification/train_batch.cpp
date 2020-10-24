/* file: train_batch.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#include "com_intel_daal_algorithms_bf_knn_classification_training_TrainingBatch.h"

#include "com/intel/daal/common_helpers.h"

#include "com_intel_daal_algorithms_bf_knn_classification_training_TrainingMethod.h"
#define defaultDense com_intel_daal_algorithms_bf_knn_classification_training_TrainingMethod_defaultDenseValue

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::bf_knn_classification::training;

/*
 * Class:     com_intel_daal_algorithms_bf_knn_classification_training_BatchTraining
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_bf_1knn_1classification_training_TrainingBatch_cInit(JNIEnv * env, jobject thisObj,
                                                                                                                jint prec, jint method)
{
    return jniBatch<bf_knn_classification::training::Method, Batch, defaultDense>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_bf_knn_classification_training_TrainingBatch
 * Method:    cInitParameter
 * Signature: (JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_bf_1knn_1classification_training_TrainingBatch_cInitParameter(JNIEnv * env,
                                                                                                                         jobject thisObj,
                                                                                                                         jlong algAddr, jint prec,
                                                                                                                         jint method)
{
    return jniBatch<bf_knn_classification::training::Method, Batch, defaultDense>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_bf_knn_classification_training_BatchTraining
 * Method:    cGetInput
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_bf_1knn_1classification_training_TrainingBatch_cGetInput(JNIEnv * env, jobject thisObj,
                                                                                                                    jlong algAddr, jint prec,
                                                                                                                    jint method)
{
    return jniBatch<bf_knn_classification::training::Method, Batch, defaultDense>::getInput(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_bf_knn_classification_training_BatchTraining
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_bf_1knn_1classification_training_TrainingBatch_cGetResult(JNIEnv * env, jobject thisObj,
                                                                                                                     jlong algAddr, jint prec,
                                                                                                                     jint method)
{
    return jniBatch<bf_knn_classification::training::Method, Batch, defaultDense>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_bf_knn_classification_training_BatchTraining
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_bf_1knn_1classification_training_TrainingBatch_cClone(JNIEnv * env, jobject thisObj,
                                                                                                                 jlong algAddr, jint prec,
                                                                                                                 jint method)
{
    return jniBatch<bf_knn_classification::training::Method, Batch, defaultDense>::getClone(prec, method, algAddr);
}

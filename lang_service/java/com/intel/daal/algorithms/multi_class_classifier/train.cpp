/* file: train.cpp */
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

/*
//++
//  JNI layer for multi_class_classifier_Train
//--
*/

#include <jni.h>
#include "JComputeMode.h"
#include "classifier/training/JTrainingResultId.h"
#include "multi_class_classifier/training/JTrainingMethod.h"
#include "classifier/training/JInputId.h"
#include "multi_class_classifier/training/JTrainingBatch.h"
#include "multi_class_classifier/training/JTrainingResult.h"
#include "daal.h"

#include "common_helpers.h"

const int innerModelID = com_intel_daal_algorithms_classifier_training_TrainingResultId_Model;

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::multi_class_classifier::training;

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_training_TrainingBatch
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_training_TrainingBatch_cInit
(JNIEnv *env, jobject obj, jint prec, jint method)
{
    return jniBatch<multi_class_classifier::training::Method, Batch, oneAgainstOne>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_training_TrainingBatch
 * Method:    cInitParameter
 * Signature:(JIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_training_TrainingBatch_cInitParameter
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method, jint cmode)
{
    return jniBatch<multi_class_classifier::training::Method, Batch, oneAgainstOne>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_training_TrainingResult
 * Method:    cGetResult
 * Signature:(JIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_training_TrainingResult_cGetResult
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method, jint cmode)
{
    return jniBatch<multi_class_classifier::training::Method, Batch, oneAgainstOne>::getResult(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_training_TrainingResult
 * Method:    cGetModel
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_training_TrainingResult_cGetModel
(JNIEnv *env, jobject obj, jlong resAddr, jint id)
{
    if (id == innerModelID)
    {
        return jniArgument<multi_class_classifier::training::Result>::
            get<classifier::training::ResultId, classifier::Model>(resAddr, classifier::training::model);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_training_TrainingBatch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_training_TrainingBatch_cClone
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multi_class_classifier::training::Method, Batch, oneAgainstOne>::getClone(prec, method, algAddr);
}

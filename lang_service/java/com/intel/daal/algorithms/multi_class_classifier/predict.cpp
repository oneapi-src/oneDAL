/* file: predict.cpp */
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
//  JNI layer for multi_class_classifier_Predict
//--
*/

#include <jni.h>
#include "JComputeMode.h"
#include "classifier/prediction/JPredictionResultId.h"
#include "classifier/prediction/JModelInputId.h"
#include "multi_class_classifier/prediction/JPredictionMethod.h"
#include "classifier/prediction/JNumericTableInputId.h"
#include "multi_class_classifier/prediction/JPredictionBatch.h"
#include "daal.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::multi_class_classifier::prediction;
using namespace daal::algorithms::multi_class_classifier::training;

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_prediction_PredictionBatch
 * Method:    cInit
 * Signature:(II)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_prediction_PredictionBatch_cInit
(JNIEnv *env, jobject obj, jint prec, jint method)
{
    return jniBatchClassifier<multi_class_classifier::prediction::Method, multi_class_classifier::training::Method,
        oneAgainstOne, multi_class_classifier::prediction::Batch, multiClassClassifierWu>::newObj(prec, method);
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_prediction_PredictionBatch
 * Method:    cInitParameter
 * Signature:(JIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_prediction_PredictionBatch_cInitParameter
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method, jint cmode)
{
    return jniBatchClassifier<multi_class_classifier::prediction::Method, multi_class_classifier::training::Method,
        oneAgainstOne, multi_class_classifier::prediction::Batch, multiClassClassifierWu>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_prediction_PredictionBatch
 * Method:    cClone
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_prediction_PredictionBatch_cClone
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method)
{
    return jniBatchClassifier<multi_class_classifier::prediction::Method, multi_class_classifier::training::Method,
        oneAgainstOne, multi_class_classifier::prediction::Batch, multiClassClassifierWu>::getClone(prec, method, algAddr);
}

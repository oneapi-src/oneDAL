/* file: predict.cpp */
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
(JNIEnv *env, jobject obj, jint prec, jint method, jlong nClasses)
{
    return jniBatchClassifier<multi_class_classifier::prediction::Method, multi_class_classifier::training::Method,
        oneAgainstOne, multi_class_classifier::prediction::Batch, multiClassClassifierWu, voteBased>::newObj(prec, method, nClasses);
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_prediction_PredictionBatch
 * Method:    cInitParameter
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_prediction_PredictionBatch_cInitParameter
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method)
{
    return jniBatchClassifier<multi_class_classifier::prediction::Method, multi_class_classifier::training::Method,
        oneAgainstOne, multi_class_classifier::prediction::Batch, multiClassClassifierWu, voteBased>::getParameter(prec, method, algAddr);
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
        oneAgainstOne, multi_class_classifier::prediction::Batch, multiClassClassifierWu, voteBased>::getClone(prec, method, algAddr);
}

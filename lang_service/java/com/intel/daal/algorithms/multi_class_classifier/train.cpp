/* file: train.cpp */
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
(JNIEnv *env, jobject obj, jint prec, jint method, jlong nClasses)
{
    return jniBatch<multi_class_classifier::training::Method, Batch, oneAgainstOne>::newObj(prec, method, nClasses);
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_training_TrainingBatch
 * Method:    cInitParameter
 * Signature:(JIII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_training_TrainingBatch_cInitParameter
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multi_class_classifier::training::Method, Batch, oneAgainstOne>::getParameter(prec, method, algAddr);
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_training_TrainingBatch
 * Method:    cGetResult
 * Signature:(JII)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_training_TrainingBatch_cGetResult
(JNIEnv *env, jobject obj, jlong algAddr, jint prec, jint method)
{
    return jniBatch<multi_class_classifier::training::Method, Batch, oneAgainstOne>::getResult(prec, method, algAddr);
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

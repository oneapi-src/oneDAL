/* file: parameter.cpp */
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

//++
//  JNI layer for multi_class_classifier_Parameter
//--


#include <jni.h>
#include "multi_class_classifier/JParameter.h"
#include "daal.h"

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Parameter
 * Method:    cSetNClasses
 * Signature:(JJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Parameter_cSetNClasses
(JNIEnv *env, jobject obj, jlong parAddr, jlong val)
{
    ((daal::algorithms::multi_class_classifier::Parameter *)parAddr)->nClasses = val;
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Parameter
 * Method:    cGetNClasses
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Parameter_cGetNClasses
(JNIEnv *env, jobject obj, jlong parAddr)
{
    return((daal::algorithms::multi_class_classifier::Parameter *)parAddr)->nClasses;
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Parameter
 * Method:    cSetMaxIterations
 * Signature:(JJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Parameter_cSetMaxIterations
(JNIEnv *env, jobject obj, jlong parAddr, jlong val)
{
    ((daal::algorithms::multi_class_classifier::Parameter *)parAddr)->maxIterations = val;
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Parameter
 * Method:    cGetMaxIterations
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Parameter_cGetMaxIterations
(JNIEnv *env, jobject obj, jlong parAddr)
{
    return((daal::algorithms::multi_class_classifier::Parameter *)parAddr)->maxIterations;
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Parameter
 * Method:    cSetAccuracyThreshold
 * Signature:(JD)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Parameter_cSetAccuracyThreshold
(JNIEnv *env, jobject obj, jlong parAddr, jdouble val)
{
    ((daal::algorithms::multi_class_classifier::Parameter *)parAddr)->accuracyThreshold = val;
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Parameter
 * Method:    cGetAccuracyThreshold
 * Signature:(J)D
 */
JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Parameter_cGetAccuracyThreshold
(JNIEnv *env, jobject obj, jlong parAddr)
{
    return((daal::algorithms::multi_class_classifier::Parameter *)parAddr)->accuracyThreshold;
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Parameter
 * Method:    cSetTraining
 * Signature:(JJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Parameter_cSetTraining
(JNIEnv *env, jobject obj, jlong parAddr, jlong trainingAddr)
{
    using namespace daal::algorithms;
    daal::services::SharedPtr<classifier::training::Batch> training =
        daal::services::staticPointerCast<classifier::training::Batch, AlgorithmIface>
            (*(daal::services::SharedPtr<AlgorithmIface> *)trainingAddr);
    ((daal::algorithms::multi_class_classifier::Parameter *)parAddr)->training = training;
}

/*
 * Class:     com_intel_daal_algorithms_multi_class_classifier_Parameter
 * Method:    cSetPrediction
 * Signature:(JJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_multi_1class_1classifier_Parameter_cSetPrediction
(JNIEnv *env, jobject obj, jlong parAddr, jlong predictionAddr)
{
    using namespace daal::algorithms;
    daal::services::SharedPtr<classifier::prediction::Batch> prediction =
        daal::services::staticPointerCast<classifier::prediction::Batch, AlgorithmIface>
            (*(daal::services::SharedPtr<AlgorithmIface> *)predictionAddr);
    ((daal::algorithms::multi_class_classifier::Parameter *)parAddr)->prediction = prediction;
}

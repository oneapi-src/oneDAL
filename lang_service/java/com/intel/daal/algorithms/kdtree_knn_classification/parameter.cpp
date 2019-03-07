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

#include <jni.h>
#include "daal.h"
#include "kdtree_knn_classification/JParameter.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter
 * Method:    cSetK
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cSetK
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong k)
{
    (*(kdtree_knn_classification::Parameter *)parAddr).k = k;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter
 * Method:    cGetK
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cGetK
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(*(kdtree_knn_classification::Parameter *)parAddr).k;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter
 * Method:    cSetSeed
 * Signature:(JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cSetSeed
(JNIEnv *env, jobject thisObj, jlong parAddr, jint seed)
{
    (*(kdtree_knn_classification::Parameter *)parAddr).seed = seed;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter
 * Method:    cGetSeed
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cGetSeed
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return(*(kdtree_knn_classification::Parameter *)parAddr).seed;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cSetEngine
(JNIEnv *env, jobject thisObj, jlong cParameter, jlong engineAddr)
{
    (((kdtree_knn_classification::Parameter *)cParameter))->engine = staticPointerCast<engines::BatchBase, AlgorithmIface> (*(SharedPtr<AlgorithmIface> *)engineAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter
 * Method:    cSetDataUseInModel
 * Signature:(JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cSetDataUseInModel
(JNIEnv *env, jobject thisObj, jlong parAddr, jint flag)
{
    (*(kdtree_knn_classification::Parameter *)parAddr).dataUseInModel = (kdtree_knn_classification::DataUseInModel)flag;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter
 * Method:    cGetDataUseInModel
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cGetDataUseInModel
(JNIEnv *env, jobject thisObj, jlong parAddr)
{
    return (jint)((*(kdtree_knn_classification::Parameter *)parAddr).dataUseInModel);
}

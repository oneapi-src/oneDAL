/* file: parameter.cpp */
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
#include "daal.h"
#include "com_intel_daal_algorithms_kdtree_knn_classification_Parameter.h"
#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES();

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_Parameter
 * Method:    cSetK
 * Signature:(JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cSetK(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                  jlong k)
{
    (*(kdtree_knn_classification::Parameter *)parAddr).k = k;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_Parameter
 * Method:    cGetK
 * Signature:(J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cGetK(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    return (*(kdtree_knn_classification::Parameter *)parAddr).k;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_Parameter
 * Method:    cSetSeed
 * Signature:(JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cSetSeed(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                                     jint seed)
{
    (*(kdtree_knn_classification::Parameter *)parAddr).seed = seed;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_Parameter
 * Method:    cGetSeed
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cGetSeed(JNIEnv * env, jobject thisObj, jlong parAddr)
{
    return (*(kdtree_knn_classification::Parameter *)parAddr).seed;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_Parameter
 * Method:    cSetEngine
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cSetEngine(JNIEnv * env, jobject thisObj,
                                                                                                       jlong cParameter, jlong engineAddr)
{
    (((kdtree_knn_classification::Parameter *)cParameter))->engine =
        staticPointerCast<engines::BatchBase, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)engineAddr);
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_Parameter
 * Method:    cSetDataUseInModel
 * Signature:(JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cSetDataUseInModel(JNIEnv * env, jobject thisObj,
                                                                                                               jlong parAddr, jint flag)
{
    (*(kdtree_knn_classification::Parameter *)parAddr).dataUseInModel = (kdtree_knn_classification::DataUseInModel)flag;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_Parameter
 * Method:    cGetDataUseInModel
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cGetDataUseInModel(JNIEnv * env, jobject thisObj,
                                                                                                               jlong parAddr)
{
    return (jint)((*(kdtree_knn_classification::Parameter *)parAddr).dataUseInModel);
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_Parameter
 * Method:    cSetVoteWeights
 * Signature:(JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cSetVoteWeights(JNIEnv * env, jobject thisObj,
                                                                                                        jlong parAddr, jint flag)
{
    (*(kdtree_knn_classification::Parameter *)parAddr).voteWeights = (kdtree_knn_classification::VoteWeights)flag;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_Parameter
 * Method:    cGetVoteWeights
 * Signature:(J)I
 */
JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cGetVoteWeights(JNIEnv * env, jobject thisObj,
                                                                                                        jlong parAddr)
{
    return (jint)((*(kdtree_knn_classification::Parameter *)parAddr).voteWeights);
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_Parameter
 * Method:    cSetResultsToCompute
 * Signature:(JI)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cSetResultsToCompute(JNIEnv * env, jobject thisObj,
                                                                                                             jlong parAddr, jlong flag)
{
    (*(kdtree_knn_classification::Parameter *)parAddr).resultsToCompute = (kdtree_knn_classification::ResultToComputeId)flag;
}

/*
 * Class:     com_intel_daal_algorithms_kdtree_knn_classification_Parameter
 * Method:    cGetResultsToCompute
 * Signature:(J)I
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kdtree_1knn_1classification_Parameter_cGetResultsToCompute(JNIEnv * env, jobject thisObj,
                                                                                                              jlong parAddr)
{
    return (jlong)((*(kdtree_knn_classification::Parameter *)parAddr).resultsToCompute);
}

/* file: parameter.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
#include "common_helpers.h"
#include "com_intel_daal_algorithms_brownboost_Parameter.h"

USING_COMMON_NAMESPACES()

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cSetWeakLearnerTraining(JNIEnv *, jobject, jlong self, jlong value)
{
    unpack<brownboost::Parameter>(self).weakLearnerTraining = unpackAlgorithm<classifier::training::Batch>(value);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cSetWeakLearnerPrediction(JNIEnv *, jobject, jlong self, jlong value)
{
    unpack<brownboost::Parameter>(self).weakLearnerPrediction = unpackAlgorithm<classifier::prediction::Batch>(value);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cSetAccuracyThreshold(JNIEnv * env, jobject thisObj, jlong self,
                                                                                                 jdouble acc)
{
    unpack<brownboost::Parameter>(self).accuracyThreshold = acc;
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cGetAccuracyThreshold(JNIEnv * env, jobject thisObj, jlong self)
{
    return (jdouble)(unpack<brownboost::Parameter>(self).accuracyThreshold);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cSetnewtonRaphsonAccuracyThreshold(JNIEnv * env, jobject thisObj,
                                                                                                              jlong self, jdouble acc)
{
    unpack<brownboost::Parameter>(self).newtonRaphsonAccuracyThreshold = acc;
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cGetnewtonRaphsonAccuracyThreshold(JNIEnv * env, jobject thisObj,
                                                                                                                 jlong self)
{
    return (jdouble)(unpack<brownboost::Parameter>(self).newtonRaphsonAccuracyThreshold);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cSetThr(JNIEnv * env, jobject thisObj, jlong self, jdouble acc)
{
    unpack<brownboost::Parameter>(self).degenerateCasesThreshold = acc;
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cGetThr(JNIEnv * env, jobject thisObj, jlong self)
{
    return (jdouble)(unpack<brownboost::Parameter>(self).degenerateCasesThreshold);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cSetMaxIterations(JNIEnv * env, jobject thisObj, jlong self, jlong nIter)
{
    unpack<brownboost::Parameter>(self).maxIterations = nIter;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cGetMaxIterations(JNIEnv * env, jobject thisObj, jlong self)
{
    return (jlong)(unpack<brownboost::Parameter>(self).maxIterations);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cSetnewtonRaphsonMaxIterations(JNIEnv * env, jobject thisObj, jlong self,
                                                                                                          jlong nIter)
{
    unpack<brownboost::Parameter>(self).newtonRaphsonMaxIterations = nIter;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_brownboost_Parameter_cGetnewtonRaphsonMaxIterations(JNIEnv * env, jobject thisObj, jlong self)
{
    return (jlong)(unpack<brownboost::Parameter>(self).newtonRaphsonMaxIterations);
}

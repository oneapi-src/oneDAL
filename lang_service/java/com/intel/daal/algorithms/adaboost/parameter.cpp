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
#include "com_intel_daal_algorithms_adaboost_Parameter.h"

USING_COMMON_NAMESPACES()

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cSetWeakLearnerTraining(JNIEnv *, jobject, jlong self, jlong value)
{
    unpack<adaboost::Parameter>(self).weakLearnerTraining = unpackAlgorithm<classifier::training::Batch>(value);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cSetWeakLearnerPrediction(JNIEnv *, jobject, jlong self, jlong value)
{
    unpack<adaboost::Parameter>(self).weakLearnerPrediction = unpackAlgorithm<classifier::prediction::Batch>(value);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cSetAccuracyThreshold(JNIEnv * env, jobject, jlong self,
                                                                                               jdouble accuracyThreshold)
{
    unpack<adaboost::Parameter>(self).accuracyThreshold = accuracyThreshold;
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cGetAccuracyThreshold(JNIEnv * env, jobject, jlong self)
{
    return (jdouble)(unpack<adaboost::Parameter>(self).accuracyThreshold);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cSetMaxIterations(JNIEnv * env, jobject, jlong self, jlong maxIterations)
{
    unpack<adaboost::Parameter>(self).maxIterations = maxIterations;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cGetMaxIterations(JNIEnv * env, jobject, jlong self)
{
    return (jlong)(unpack<adaboost::Parameter>(self).maxIterations);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cSetLearningRate(JNIEnv * env, jobject, jlong self, jlong learningRate)
{
    unpack<adaboost::Parameter>(self).learningRate = learningRate;
}

JNIEXPORT double JNICALL Java_com_intel_daal_algorithms_adaboost_Parameter_cGetLearningRate(JNIEnv * env, jobject, jlong self)
{
    return (double)(unpack<adaboost::Parameter>(self).learningRate);
}

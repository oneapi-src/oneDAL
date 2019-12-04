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
#include "com_intel_daal_algorithms_logitboost_Parameter.h"

USING_COMMON_NAMESPACES()

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cSetWeakLearnerTraining(JNIEnv *, jobject, jlong self, jlong value)
{
    unpack<logitboost::Parameter>(self).weakLearnerTraining = unpackAlgorithm<regression::training::Batch>(value);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cSetWeakLearnerPrediction(JNIEnv *, jobject, jlong self, jlong value)
{
    unpack<logitboost::Parameter>(self).weakLearnerPrediction = unpackAlgorithm<regression::prediction::Batch>(value);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cSetAccuracyThreshold(JNIEnv * env, jobject, jlong self, jdouble acc)
{
    unpack<logitboost::Parameter>(self).accuracyThreshold = acc;
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cGetAccuracyThreshold(JNIEnv * env, jobject, jlong self)
{
    return (jdouble)(unpack<logitboost::Parameter>(self).accuracyThreshold);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cSetWeightsThreshold(JNIEnv * env, jobject, jlong self, jdouble acc)
{
    unpack<logitboost::Parameter>(self).weightsDegenerateCasesThreshold = acc;
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cGetWeightsThreshold(JNIEnv * env, jobject, jlong self)
{
    return (jdouble)(unpack<logitboost::Parameter>(self).weightsDegenerateCasesThreshold);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cSetResponsesThreshold(JNIEnv * env, jobject, jlong self, jdouble acc)
{
    unpack<logitboost::Parameter>(self).responsesDegenerateCasesThreshold = acc;
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cGetResponsesThreshold(JNIEnv * env, jobject, jlong self)
{
    return (jdouble)(unpack<logitboost::Parameter>(self).responsesDegenerateCasesThreshold);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cSetMaxIterations(JNIEnv * env, jobject, jlong self, jlong nIter)
{
    unpack<logitboost::Parameter>(self).maxIterations = nIter;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cGetMaxIterations(JNIEnv * env, jobject, jlong self)
{
    return (jlong)(unpack<logitboost::Parameter>(self).maxIterations);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cSetNClasses(JNIEnv * env, jobject, jlong self, jlong nIter)
{
    unpack<logitboost::Parameter>(self).nClasses = nIter;
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_logitboost_Parameter_cGetNClasses(JNIEnv * env, jobject, jlong self)
{
    return (jlong)(unpack<logitboost::Parameter>(self).nClasses);
}

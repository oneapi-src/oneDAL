/* file: model.cpp */
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
#include "com_intel_daal_algorithms_brownboost_Model.h"

USING_COMMON_NAMESPACES()

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_brownboost_Model_cGetNumberOfWeakLearners(JNIEnv *, jobject, jlong self)
{
    return (jlong)(unpackModel<brownboost::Model>(self)->getNumberOfWeakLearners());
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_brownboost_Model_cGetWeakLearnerModel(JNIEnv *, jobject, jlong self, jlong idx)
{
    return packModel(unpackModel<brownboost::Model>(self)->getWeakLearnerModel((size_t)idx));
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_brownboost_Model_cAddWeakLearnerModel(JNIEnv *, jobject, jlong self, jlong model)
{
    unpackModel<brownboost::Model>(self)->addWeakLearnerModel(unpackModel<classifier::Model>(model));
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_brownboost_Model_cClearWeakLearnerModels(JNIEnv *, jobject, jlong self)
{
    unpackModel<brownboost::Model>(self)->clearWeakLearnerModels();
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_brownboost_Model_cGetNumberOfFeatures(JNIEnv *, jobject, jlong self)
{
    return (jlong)(unpackModel<brownboost::Model>(self)->getNumberOfFeatures());
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_brownboost_Model_cGetAlpha(JNIEnv *, jobject, jlong self)
{
    return packTable(unpackModel<brownboost::Model>(self)->getAlpha());
}

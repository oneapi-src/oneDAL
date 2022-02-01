/* file: model.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
#include "com/intel/daal/common_helpers.h"
#include "com_intel_daal_algorithms_stump_classification_Model.h"

using namespace daal;
using namespace daal::algorithms;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_stump_classification_Model_cGetSplitFeature(JNIEnv *, jobject, jlong self)
{
    return (jlong)(unpackModel<stump::classification::Model>(self)->getSplitFeature());
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_stump_classification_Model_cGetSplitValue(JNIEnv *, jobject, jlong self)
{
    return (jdouble)(unpackModel<stump::classification::Model>(self)->getSplitValue<double>());
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_stump_classification_Model_cGetLeftValue(JNIEnv *, jobject, jlong self)
{
    return (jdouble)(unpackModel<stump::classification::Model>(self)->getLeftValue<double>());
}

JNIEXPORT jdouble JNICALL Java_com_intel_daal_algorithms_stump_classification_Model_cGetRightValue(JNIEnv *, jobject, jlong self)
{
    return (jdouble)(unpackModel<stump::classification::Model>(self)->getRightValue<double>());
}

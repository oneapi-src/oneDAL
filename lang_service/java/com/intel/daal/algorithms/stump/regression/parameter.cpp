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
#include "com_intel_daal_algorithms_stump_regression_Parameter.h"

using namespace daal;
using namespace daal::algorithms;

JNIEXPORT jint JNICALL Java_com_intel_daal_algorithms_stump_regression_Parameter_cGetVarImportance(JNIEnv *, jobject, jlong self)
{
    return (jint)(unpack<stump::regression::Parameter>(self).varImportance);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_stump_regression_Parameter_cSetVarImportance(JNIEnv *, jobject, jlong self, jint value)
{
    unpack<stump::regression::Parameter>(self).varImportance = (stump::regression::VariableImportanceMode)(value);
}

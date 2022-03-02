/* file: result.cpp */
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
#include "com_intel_daal_algorithms_distributions_Result.h"

#include "daal.h"

#include "com/intel/daal/common_helpers.h"

#include "com_intel_daal_algorithms_distributions_ResultId.h"
#define randomNumbers com_intel_daal_algorithms_distributions_ResultId_randomNumbersId

USING_COMMON_NAMESPACES()
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_distributions_Result
 * Method:    cGetValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_distributions_Result_cGetValue(JNIEnv * env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<distributions::Result>::get<distributions::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_distributions_Result
 * Method:    cSetValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_distributions_Result_cSetValue(JNIEnv * env, jobject thisObj, jlong resAddr, jint id,
                                                                                     jlong tensorAddr)
{
    if (id == randomNumbers)
    {
        jniArgument<distributions::Result>::set<distributions::ResultId, NumericTable>(resAddr, id, tensorAddr);
    }
}

/* file: optional_argument.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

#include <jni.h>/* Header for class com_intel_daal_algorithms_Result */
#include "JOptionalArgument.h"

#include "daal.h"

using namespace daal;
/*
* Class:     com_intel_daal_algorithms_OptionalArgument
* Method:    cNewOptionalArgument
* Signature: ()J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_OptionalArgument_cNewOptionalArgument
(JNIEnv *, jobject, jlong size)
{
    algorithms::OptionalArgumentPtr pArg(new algorithms::OptionalArgument(size));
    data_management::SerializationIfacePtr *resultShPtr = new data_management::SerializationIfacePtr(pArg);
    return (jlong)resultShPtr;
}

/*
* Class:     com_intel_daal_algorithms_OptionalArgument
* Method:    cGetValue
* Signature: (JJ)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_OptionalArgument_cGetValue
(JNIEnv *env, jobject thisObj, jlong argAddr, jlong idx)
{
    data_management::SerializationIfacePtr pArg = *(data_management::SerializationIfacePtr*)argAddr;
    data_management::SerializationIfacePtr ptr = static_cast<algorithms::OptionalArgument*>(pArg.get())->get(idx);
    return (jlong)new data_management::SerializationIfacePtr(ptr);
}

/*
* Class:     com_intel_daal_algorithms_OptionalArgument
* Method:    cSetValue
* Signature: (JJJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_OptionalArgument_cSetValue
(JNIEnv *, jobject thisObj, jlong argAddr, jlong valueAddr, jlong idx)
{
    data_management::SerializationIfacePtr pArg = *(data_management::SerializationIfacePtr*)argAddr;
    static_cast<algorithms::OptionalArgument*>(pArg.get())->set(idx, *((data_management::SerializationIfacePtr *)valueAddr));
}

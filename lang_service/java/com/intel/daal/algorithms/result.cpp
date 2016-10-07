/* file: result.cpp */
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
#include "JResult.h"

#include "daal.h"

/*
 * Class:     com_intel_daal_algorithms_Result
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_Result_cDispose
(JNIEnv *env, jobject thisObj, jlong resAddr)
{
    delete (daal::data_management::SerializationIfacePtr *)resAddr;
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_Result_cCheckPartRes
(JNIEnv *env, jobject thisObj, jlong resAddr, jlong partResAddr, jlong parAddr, jint method)
{
    daal::algorithms::Result *resPtr =
        (daal::algorithms::Result *)(((daal::data_management::SerializationIfacePtr *)resAddr)->get());
    daal::algorithms::PartialResult *partResPtr =
                            (daal::algorithms::PartialResult *)
                            (((daal::data_management::SerializationIfacePtr *)partResAddr)->get());

    resPtr->check(partResPtr,(daal::algorithms::Parameter*)parAddr,method);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_Result_cCheckInput
(JNIEnv *env, jobject thisObj, jlong resAddr, jlong inputAddr, jlong parAddr, jint method)
{
    daal::algorithms::Result *resPtr =
        (daal::algorithms::Result *)
        (((daal::data_management::SerializationIfacePtr *)resAddr)->get());
    resPtr->check((daal::algorithms::Input*)inputAddr,(daal::algorithms::Parameter*)parAddr,(int)method);
}

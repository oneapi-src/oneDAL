/* file: result.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
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

/* file: partial_result.cpp */
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

#include "daal.h"
#include "JPartialResult.h"


JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PartialResult_cCheck
(JNIEnv *env, jobject thisObj, jlong partResAddr, jlong parAddr, jint method)
{
    daal::algorithms::PartialResult *partResPtr =
                            (daal::algorithms::PartialResult *)
                            (((daal::data_management::SerializationIfacePtr *)partResAddr)->get());
    partResPtr->check((daal::algorithms::Parameter*)parAddr,method);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_PartialResult_cCheckInput
(JNIEnv *env, jobject thisObj, jlong partResAddr, jlong inputAddr, jlong parAddr, jint method)
{
    daal::algorithms::PartialResult *partResPtr =
                            (daal::algorithms::PartialResult *)
                            (((daal::data_management::SerializationIfacePtr *)partResAddr)->get());
    partResPtr->check((daal::algorithms::Input*)inputAddr,(daal::algorithms::Parameter*)parAddr,method);
}

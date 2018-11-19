/* file: input.cpp */
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
#include "JInput.h"

/*
 * Class:     com_intel_daal_algorithms_Result
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_Input_cCheck
(JNIEnv *env, jobject thisObj, jlong inputAddr, jlong parAddr, jint method)
{
    ((daal::algorithms::Input *)inputAddr)->check((daal::algorithms::Parameter*)parAddr,method);
}

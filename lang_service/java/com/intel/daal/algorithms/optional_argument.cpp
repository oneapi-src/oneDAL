/* file: optional_argument.cpp */
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

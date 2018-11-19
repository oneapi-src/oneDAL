/* file: training_init_partial_result_base.cpp */
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

#include <jni.h>

#include "daal.h"

#include "implicit_als/training/init/JInitPartialResultBase.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::implicit_als::training::init;

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitPartialResultBase
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitPartialResultBase_cNewPartialResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<PartialResultBase>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitPartialResultBase
 * Method:    cGetPartialResultCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitPartialResultBase_cGetPartialResultCollection
(JNIEnv *env, jobject thisObj, jlong partialResultAddr, jint id)
{
    return jniArgument<PartialResultBase>::
        get<PartialResultBaseId, KeyValueDataCollection>(partialResultAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitPartialResultBase
 * Method:    cSetPartialResultCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitPartialResultBase_cSetPartialResultCollection
(JNIEnv *env, jobject thisObj, jlong partialResultAddr, jint id, jlong collectionAddr)
{
    jniArgument<PartialResultBase>::
        set<PartialResultBaseId, KeyValueDataCollection>(partialResultAddr, id, collectionAddr);
}

/*
 * Class:     com_intel_daal_algorithms_implicit_als_training_init_InitPartialResultBase
 * Method:    cGetPartialResultTable
 * Signature: (JIJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_implicit_1als_training_init_InitPartialResultBase_cGetPartialResultTable
(JNIEnv *env, jobject thisObj, jlong partialResultAddr, jint id, jlong key)
{
    return jniArgument<PartialResultBase>::
        get<PartialResultBaseId, NumericTable>(partialResultAddr, id, key);
}

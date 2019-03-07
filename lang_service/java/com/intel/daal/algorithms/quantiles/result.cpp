/* file: result.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
#include "quantiles/JResult.h"
#include "quantiles/JMethod.h"

#include "common_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::quantiles;

/*
 * Class:     com_intel_daal_algorithms_quantiles_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_quantiles_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<quantiles::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_quantiles_Result
 * Method:    cGetResultTable
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_quantiles_Result_cGetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<quantiles::Result>::get<ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_quantiles_Result
 * Method:    cSetResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_quantiles_Result_cSetResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<quantiles::Result>::set<ResultId, NumericTable>(resAddr, id, ntAddr);
}

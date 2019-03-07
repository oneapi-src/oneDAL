/* file: batch_result.cpp */
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
#include "normalization/minmax/JMethod.h"
#include "normalization/minmax/JResult.h"
#include "normalization/minmax/JResultId.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::normalization::minmax;

/*
 * Class:     com_intel_daal_algorithms_normalization_minmax_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_minmax_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<normalization::minmax::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_normalization_minmax_Result
 * Method:    cGetNormalizedData
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_minmax_Result_cGetNormalizedData
(JNIEnv *env, jobject thisObj, jlong resAddr)
{
    return jniArgument<normalization::minmax::Result>::
        get<normalization::minmax::ResultId, NumericTable>(resAddr, normalization::minmax::normalizedData);
}

/*
 * Class:     com_intel_daal_algorithms_normalization_minmax_Result
 * Method:    cSetNormalizedData
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_normalization_minmax_Result_cSetNormalizedData
(JNIEnv *env, jobject thisObj, jlong resAddr, jlong ntAddr)
{
    jniArgument<normalization::minmax::Result>::
        set<normalization::minmax::ResultId, NumericTable>(resAddr, normalization::minmax::normalizedData, ntAddr);
}

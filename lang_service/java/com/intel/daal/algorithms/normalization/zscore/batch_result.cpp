/* file: batch_result.cpp */
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
#include "normalization/zscore/JMethod.h"
#include "normalization/zscore/JResult.h"
#include "normalization/zscore/JResultId.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::normalization::zscore;

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Result
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Result_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<normalization::zscore::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Result
 * Method:    cGetResultNumericTable
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Result_cGetResultNumericTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<normalization::zscore::Result>::
        get<normalization::zscore::ResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_normalization_zscore_Result
 * Method:    cSetResultNumericTable
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_normalization_zscore_Result_cSetResultNumericTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<normalization::zscore::Result>::
        set<normalization::zscore::ResultId, NumericTable>(resAddr, id, ntAddr);
}

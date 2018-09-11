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
#include "pca/transform/JTransformMethod.h"
#include "pca/transform/JTransformResult.h"
#include "pca/transform/JTransformResultId.h"

#include "daal.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::pca::transform;

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformResult
 * Method:    cNewResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformResult_cNewResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<pca::transform::Result>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformResult
 * Method:    cGetTransformedData
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformResult_cGetTransformedData
(JNIEnv *env, jobject thisObj, jlong resAddr)
{
    return jniArgument<pca::transform::Result>::
        get<pca::transform::ResultId, NumericTable>(resAddr, pca::transform::transformedData);
}

/*
 * Class:     com_intel_daal_algorithms_pca_transform_TransformResult
 * Method:    cSetTransformedData
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_transform_TransformResult_cSetTransformedData
(JNIEnv *env, jobject thisObj, jlong resAddr, jlong ntAddr)
{
    jniArgument<pca::transform::Result>::
        set<pca::transform::ResultId, NumericTable>(resAddr, pca::transform::transformedData, ntAddr);
}

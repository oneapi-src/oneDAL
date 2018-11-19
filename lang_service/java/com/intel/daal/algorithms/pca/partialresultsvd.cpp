/* file: partialresultsvd.cpp */
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

#include "pca/JPartialSVDResult.h"
#include "pca/JPartialSVDTableResultID.h"
#include "pca/JPartialSVDCollectionResultID.h"

#include "pca/JMethod.h"
#include "common_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::pca;

#define jnObservationsId      com_intel_daal_algorithms_pca_PartialSVDTableResultID_nObservationsId
#define jsumSVDId             com_intel_daal_algorithms_pca_PartialSVDTableResultID_sumSVDId
#define jsumSquaresSVDId      com_intel_daal_algorithms_pca_PartialSVDTableResultID_sumSquaresSVDId
#define jsvdAuxiliaryDataId   com_intel_daal_algorithms_pca_PartialSVDCollectionResultID_svdAuxiliaryDataId

/*
 * Class:     com_intel_daal_algorithms_pca_PartialSVDResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_PartialSVDResult_cNewPartialResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<pca::PartialResult<svdDense> >::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_pca_PartialSVDResult
 * Method:    cGetPartialResultValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_PartialSVDResult_cGetPartialResultValue
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if(id == jnObservationsId || id == jsumSVDId || id == jsumSquaresSVDId)
    {
        return jniArgument<pca::PartialResult<svdDense> >::get<PartialSVDTableResultId, NumericTable>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_pca_PartialSVDResult
 * Method:    cSetPartialResultValue
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_PartialSVDResult_cSetPartialResultValue
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    if(id == jnObservationsId || id == jsumSVDId || id == jsumSquaresSVDId)
    {
        jniArgument<pca::PartialResult<svdDense> >::set<PartialSVDTableResultId, NumericTable>(resAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_pca_PartialSVDResult
 * Method:    cGetPartialResultCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_PartialSVDResult_cGetPartialResultCollection
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    if(id == jsvdAuxiliaryDataId)
    {
        return jniArgument<pca::PartialResult<svdDense> >::get<PartialSVDCollectionResultId, DataCollection>(resAddr, id);
    }

    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_pca_PartialSVDResult
 * Method:    cSetPartialResultCollection
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_PartialSVDResult_cSetPartialResultCollection
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong dcAddr)
{
    if(id == jsvdAuxiliaryDataId)
    {
        jniArgument<pca::PartialResult<svdDense> >::set<PartialSVDCollectionResultId, DataCollection>(resAddr, id, dcAddr);
    }
}

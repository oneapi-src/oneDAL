/* file: partialresultsvd.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <jni.h>
#include "daal.h"

#include "com_intel_daal_algorithms_pca_PartialSVDResult.h"

#include "com_intel_daal_algorithms_pca_Method.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::pca;

#include "com_intel_daal_algorithms_pca_PartialSVDTableResultID.h"
#define jnObservationsId com_intel_daal_algorithms_pca_PartialSVDTableResultID_nObservationsId
#define jsumSVDId        com_intel_daal_algorithms_pca_PartialSVDTableResultID_sumSVDId
#define jsumSquaresSVDId com_intel_daal_algorithms_pca_PartialSVDTableResultID_sumSquaresSVDId
#include "com_intel_daal_algorithms_pca_PartialSVDCollectionResultID.h"
#define jsvdAuxiliaryDataId com_intel_daal_algorithms_pca_PartialSVDCollectionResultID_svdAuxiliaryDataId

/*
 * Class:     com_intel_daal_algorithms_pca_PartialSVDResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_PartialSVDResult_cNewPartialResult(JNIEnv * env, jobject thisObj)
{
    return jniArgument<pca::PartialResult<svdDense> >::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_pca_PartialSVDResult
 * Method:    cGetPartialResultValue
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_PartialSVDResult_cGetPartialResultValue(JNIEnv * env, jobject thisObj, jlong resAddr,
                                                                                                   jint id)
{
    if (id == jnObservationsId || id == jsumSVDId || id == jsumSquaresSVDId)
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
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_PartialSVDResult_cSetPartialResultValue(JNIEnv * env, jobject thisObj, jlong resAddr,
                                                                                                  jint id, jlong ntAddr)
{
    if (id == jnObservationsId || id == jsumSVDId || id == jsumSquaresSVDId)
    {
        jniArgument<pca::PartialResult<svdDense> >::set<PartialSVDTableResultId, NumericTable>(resAddr, id, ntAddr);
    }
}

/*
 * Class:     com_intel_daal_algorithms_pca_PartialSVDResult
 * Method:    cGetPartialResultCollection
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_PartialSVDResult_cGetPartialResultCollection(JNIEnv * env, jobject thisObj, jlong resAddr,
                                                                                                        jint id)
{
    if (id == jsvdAuxiliaryDataId)
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
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_PartialSVDResult_cSetPartialResultCollection(JNIEnv * env, jobject thisObj, jlong resAddr,
                                                                                                       jint id, jlong dcAddr)
{
    if (id == jsvdAuxiliaryDataId)
    {
        jniArgument<pca::PartialResult<svdDense> >::set<PartialSVDCollectionResultId, DataCollection>(resAddr, id, dcAddr);
    }
}

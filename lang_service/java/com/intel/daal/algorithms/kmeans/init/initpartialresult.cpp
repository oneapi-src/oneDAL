/* file: initpartialresult.cpp */
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
#include "kmeans/init/JInitPartialResult.h"
#include "kmeans/init/JInitDistributedStep2LocalPlusPlusPartialResult.h"
#include "kmeans/init/JInitDistributedStep4LocalPlusPlusPartialResult.h"
#include "kmeans/init/JInitDistributedStep3MasterPlusPlusPartialResult.h"
#include "kmeans/init/JInitDistributedStep5MasterPlusPlusPartialResult.h"

#include "common_defines.i"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::kmeans::init;

/*
 * Class:     com_intel_daal_algorithms_kmeans_PartialResult
 * Method:    cNewPartialResult
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitPartialResult_cNewPartialResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<kmeans::init::PartialResult>::newObj();
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_PartialResult
 * Method:    cGetPartialResultTable
 * Signature:(JI)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitPartialResult_cGetPartialResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<kmeans::init::PartialResult>::get<kmeans::init::PartialResultId, NumericTable>(resAddr, id);
}

/*
 * Class:     com_intel_daal_algorithms_kmeans_PartialResult
 * Method:    cSetPartialResultTable
 * Signature: (JIJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitPartialResult_cSetPartialResultTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<kmeans::init::PartialResult>::set<kmeans::init::PartialResultId, NumericTable>(resAddr, id, ntAddr);
}

/////////////////////////////////////// plusPlus methods ///////////////////////////////////////////////////////
///////////////////////////////////////   step2Local     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusPartialResult
* Method:    cNewPartialResult
* Signature: ()J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusPartialResult_cNewPartialResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<kmeans::init::DistributedStep2LocalPlusPlusPartialResult>::newObj();
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusPartialResult
* Method:    cSetTable
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusPartialResult_cSetTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<kmeans::init::DistributedStep2LocalPlusPlusPartialResult>::
        set<kmeans::init::DistributedStep2LocalPlusPlusPartialResultId, NumericTable>(resAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusPartialResult
* Method:    cGetTable
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusPartialResult_cGetTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<kmeans::init::DistributedStep2LocalPlusPlusPartialResult>::
        get<kmeans::init::DistributedStep2LocalPlusPlusPartialResultId, NumericTable>(resAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusPartialResult
* Method:    cSetDataCollection
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusPartialResult_cSetDataCollection
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong addr)
{
    jniArgument<kmeans::init::DistributedStep2LocalPlusPlusPartialResult>::
        set<kmeans::init::DistributedStep2LocalPlusPlusPartialResultDataId, DataCollection>(resAddr, id, addr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusPartialResult
* Method:    cGetDataCollection
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusPartialResult_cGetDataCollection
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<kmeans::init::DistributedStep2LocalPlusPlusPartialResult>::
        get<kmeans::init::DistributedStep2LocalPlusPlusPartialResultDataId, DataCollection>(resAddr, id);
}

///////////////////////////////////////   step3Master     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep3MasterPlusPlusPartialResult
* Method:    cNewPartialResult
* Signature: ()J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep3MasterPlusPlusPartialResult_cNewPartialResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<kmeans::init::DistributedStep3MasterPlusPlusPartialResult>::newObj();
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep3MasterPlusPlusPartialResult
* Method:    cGetTable
* Signature: (JII)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep3MasterPlusPlusPartialResult_cGetTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jint key)
{
    return jniArgument<kmeans::init::DistributedStep3MasterPlusPlusPartialResult>::
        get<kmeans::init::DistributedStep3MasterPlusPlusPartialResultId, NumericTable>(resAddr, id, key);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep3MasterPlusPlusPartialResult
* Method:    cGetKeyValueDataCollection
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep3MasterPlusPlusPartialResult_cGetKeyValueDataCollection
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<kmeans::init::DistributedStep3MasterPlusPlusPartialResult>::
        get<kmeans::init::DistributedStep3MasterPlusPlusPartialResultId, KeyValueDataCollection>(resAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep3MasterPlusPlusPartialResult
* Method:    cGetSerializableBase
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep3MasterPlusPlusPartialResult_cGetSerializableBase
(JNIEnv *, jobject, jlong resAddr, jint id)
{
    return jniArgument<kmeans::init::DistributedStep3MasterPlusPlusPartialResult>::
        get<kmeans::init::DistributedStep3MasterPlusPlusPartialResultDataId, SerializationIface>(resAddr, id);
}

///////////////////////////////////////   step4Local     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusPartialResult
* Method:    cNewPartialResult
* Signature: ()J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusPartialResult_cNewPartialResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<kmeans::init::DistributedStep4LocalPlusPlusPartialResult>::newObj();
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusPartialResult
* Method:    cSetTable
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusPartialResult_cSetTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<kmeans::init::DistributedStep4LocalPlusPlusPartialResult>::
        set<kmeans::init::DistributedStep4LocalPlusPlusPartialResultId, NumericTable>(resAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusPartialResult
* Method:    cGetTable
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusPartialResult_cGetTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<kmeans::init::DistributedStep4LocalPlusPlusPartialResult>::
        get<kmeans::init::DistributedStep4LocalPlusPlusPartialResultId, NumericTable>(resAddr, id);
}

///////////////////////////////////////   step5Master     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusPartialResult
* Method:    cNewPartialResult
* Signature: ()J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusPartialResult_cNewPartialResult
(JNIEnv *env, jobject thisObj)
{
    return jniArgument<kmeans::init::DistributedStep5MasterPlusPlusPartialResult>::newObj();
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusPartialResult
* Method:    cSetTable
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusPartialResult_cSetTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id, jlong ntAddr)
{
    jniArgument<kmeans::init::DistributedStep5MasterPlusPlusPartialResult>::
        set<kmeans::init::DistributedStep5MasterPlusPlusPartialResultId, NumericTable>(resAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusPartialResult
* Method:    cGetTable
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusPartialResult_cGetTable
(JNIEnv *env, jobject thisObj, jlong resAddr, jint id)
{
    return jniArgument<kmeans::init::DistributedStep5MasterPlusPlusPartialResult>::
        get<kmeans::init::DistributedStep5MasterPlusPlusPartialResultId, NumericTable>(resAddr, id);
}

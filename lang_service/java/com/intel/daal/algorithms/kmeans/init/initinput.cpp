/* file: initinput.cpp */
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
#include "common_defines.i"
#include "kmeans/init/JInitInput.h"
#include "kmeans/init/JInitDistributedStep2LocalPlusPlusInput.h"
#include "kmeans/init/JInitDistributedStep4LocalPlusPlusInput.h"
#include "kmeans/init/JInitDistributedStep3MasterPlusPlusInput.h"
#include "kmeans/init/JInitDistributedStep5MasterPlusPlusInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES();
using namespace daal::algorithms::kmeans::init;

/*
* Class:     com_intel_daal_algorithms_kmeans_Input
* Method:    cSetData
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitInput_cSetData
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<kmeans::init::Input>::set<kmeans::init::InputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_Input
* Method:    cGetData
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitInput_cGetData
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<kmeans::init::Input>::get<kmeans::init::InputId, NumericTable>(inputAddr, id);
}

/////////////////////////////////////// plusPlus methods ///////////////////////////////////////////////////////
///////////////////////////////////////   step2Local     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusInput
* Method:    cSetTable
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusInput_cSetTable
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<kmeans::init::DistributedStep2LocalPlusPlusInput>::
        set<kmeans::init::DistributedStep2LocalPlusPlusInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusInput
* Method:    cGetTable
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusInput_cGetTable
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<kmeans::init::DistributedStep2LocalPlusPlusInput>::
        get<kmeans::init::DistributedStep2LocalPlusPlusInputId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusInput
* Method:    cSetDataCollection
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusInput_cSetDataCollection
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong addr)
{
    jniInput<kmeans::init::DistributedStep2LocalPlusPlusInput>::
        set<kmeans::init::DistributedLocalPlusPlusInputDataId, DataCollection>(inputAddr, id, addr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusInput
* Method:    cGetDataCollection
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep2LocalPlusPlusInput_cGetDataCollection
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<kmeans::init::DistributedStep2LocalPlusPlusInput>::
        get<kmeans::init::DistributedLocalPlusPlusInputDataId, DataCollection>(inputAddr, id);
}

///////////////////////////////////////   step3Master     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep3MasterPlusPlusInput
* Method:    cAddInput
* Signature: (JIIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep3MasterPlusPlusInput_cAddInput
(JNIEnv *, jobject, jlong inputAddr, jint id, jint key, jlong addr)
{
    jniInput<kmeans::init::DistributedStep3MasterPlusPlusInput>::
        add<kmeans::init::DistributedStep3MasterPlusPlusInputId, NumericTable>(inputAddr, id, key, addr);
}

///////////////////////////////////////   step4Local     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusInput
* Method:    cSetTable
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusInput_cSetTable
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<kmeans::init::DistributedStep4LocalPlusPlusInput>::
        set<kmeans::init::DistributedStep4LocalPlusPlusInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusInput
* Method:    cGetTable
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusInput_cGetTable
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<kmeans::init::DistributedStep4LocalPlusPlusInput>::
        get<kmeans::init::DistributedStep4LocalPlusPlusInputId, NumericTable>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusInput
* Method:    cSetDataCollection
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusInput_cSetDataCollection
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong addr)
{
    jniInput<kmeans::init::DistributedStep4LocalPlusPlusInput>::
        set<kmeans::init::DistributedLocalPlusPlusInputDataId, DataCollection>(inputAddr, id, addr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusInput
* Method:    cGetDataCollection
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep4LocalPlusPlusInput_cGetDataCollection
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<kmeans::init::DistributedStep4LocalPlusPlusInput>::
        get<kmeans::init::DistributedLocalPlusPlusInputDataId, DataCollection>(inputAddr, id);
}

///////////////////////////////////////   step5Master     ///////////////////////////////////////////////////////
/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusInput
* Method:    cAddInput
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusInput_cAddInput
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<kmeans::init::DistributedStep5MasterPlusPlusInput>::
        add<kmeans::init::DistributedStep5MasterPlusPlusInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusInput
* Method:    cGetInput
* Signature: (JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusInput_cGetInput
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<kmeans::init::DistributedStep5MasterPlusPlusInput>::
        get<kmeans::init::DistributedStep5MasterPlusPlusInputDataId, SerializationIface>(inputAddr, id);
}

/*
* Class:     com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusInput
* Method:    cSetInput
* Signature: (JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_init_InitDistributedStep5MasterPlusPlusInput_cSetInput
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong addr)
{
    jniInput<kmeans::init::DistributedStep5MasterPlusPlusInput>::
        set<kmeans::init::DistributedStep5MasterPlusPlusInputDataId, SerializationIface>(inputAddr, id, addr);
}

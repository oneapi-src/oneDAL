/* file: init_distributed_step2_master_input.cpp */
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
#include "com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep2MasterInput.h"

#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::gbt::regression::init;

/*
* Class:     com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep2MasterInput
* Method:    cSetDataCollection
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep2MasterInput_cSetDataCollection
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong dcAddr)
{
    jniInput<DistributedInput<step2Master> >::set<Step2MasterCollectionInputId, DataCollection>(inputAddr, id, dcAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep2MasterInput
* Method:    cAddNumericTable
* Signature:(JIJ)V
*/
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep2MasterInput_cAddNumericTable
(JNIEnv *, jobject, jlong inputAddr, jint id, jlong ntAddr)
{
    jniInput<DistributedInput<step2Master> >::add<Step2MasterCollectionInputId, NumericTable>(inputAddr, id, ntAddr);
}

/*
* Class:     com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep2MasterInput
* Method:    cGetDataCollection
* Signature:(JI)J
*/
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_gbt_regression_init_InitDistributedStep2MasterInput_cGetDataCollection
(JNIEnv *, jobject, jlong inputAddr, jint id)
{
    return jniInput<DistributedInput<step2Master> >::get<Step2MasterCollectionInputId, DataCollection>(inputAddr, id);
}

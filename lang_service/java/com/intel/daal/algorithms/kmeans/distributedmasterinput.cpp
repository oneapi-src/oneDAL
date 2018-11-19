/* file: distributedmasterinput.cpp */
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
#include "kmeans/JDistributedStep2MasterInput.h"
#include "common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::kmeans;


/*
 * Class:     com_intel_daal_algorithms_kmeans_DistributedStep2MasterInput
 * Method:    cAddInput
 * Signature: (JIJ)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_kmeans_DistributedStep2MasterInput_cAddInput
(JNIEnv *env, jobject thisObj, jlong inputAddr, jint id, jlong partialResultAddr)
{
    jniInput<kmeans::DistributedStep2MasterInput>::add<kmeans::MasterInputId, kmeans::PartialResult>(inputAddr, id, partialResultAddr);
}

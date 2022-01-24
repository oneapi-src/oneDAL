/* file: distributed_step12_local.cpp */
/*******************************************************************************
* Copyright 2014-2022 Intel Corporation
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
#include "com_intel_daal_algorithms_dbscan_DistributedStep12Local.h"

#include "com/intel/daal/common_helpers.h"

USING_COMMON_NAMESPACES()
using namespace daal::algorithms::dbscan;

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep12Local_cInit(JNIEnv *, jobject, jint prec, jint method,
                                                                                           jlong blockIndex, jlong nBlocks)
{
    return jniDistributed<step12Local, dbscan::Method, Distributed, defaultDense>::newObj(prec, method, blockIndex, nBlocks);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep12Local_cInitParameter(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                    jint prec, jint method)
{
    return jniDistributed<step12Local, dbscan::Method, Distributed, defaultDense>::getBaseParameter(prec, method, algAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep12Local_cGetInput(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                               jint prec, jint method)
{
    return jniDistributed<step12Local, dbscan::Method, Distributed, defaultDense>::getInput(prec, method, algAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep12Local_cGetPartialResult(JNIEnv * env, jobject thisObj, jlong algAddr,
                                                                                                       jint prec, jint method)
{
    return jniDistributed<step12Local, dbscan::Method, Distributed, defaultDense>::getPartialResult(prec, method, algAddr);
}

JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep12Local_cSetPartialResult(JNIEnv *, jobject, jlong algAddr, jint prec,
                                                                                                      jint method, jlong partialResultAddr)
{
    jniDistributed<step12Local, dbscan::Method, Distributed, defaultDense>::setPartialResult<dbscan::DistributedPartialResultStep12>(
        prec, method, algAddr, partialResultAddr);
}

JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_dbscan_DistributedStep12Local_cClone(JNIEnv * env, jobject thisObj, jlong algAddr, jint prec,
                                                                                            jint method)
{
    return jniDistributed<step12Local, dbscan::Method, Distributed, defaultDense>::getClone(prec, method, algAddr);
}

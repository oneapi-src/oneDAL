/* file: analysis_online.cpp */
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

#include "com_intel_daal_algorithms_AnalysisOnline.h"
#include "services/daal_defines.h"
#include "algorithms/algorithm.h"
#include "com/intel/daal/common_helpers_functions.h"

using namespace daal;
using namespace daal::services;
using namespace daal::algorithms;

/*
 * Class:     com_intel_daal_algorithms_AnalysisOffline
 * Method:    cCompute
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisOnline_cCompute(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Analysis<online> > alg = staticPointerCast<Analysis<online>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->compute());
}

/*
 * Class:     com_intel_daal_algorithms_AnalysisOnline
 * Method:    cFinalizeCompute
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisOnline_cFinalizeCompute(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Analysis<online> > alg = staticPointerCast<Analysis<online>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->finalizeCompute());
}

/*
 * Class:     com_intel_daal_algorithms_AnalysisOnline
 * Method:    cCheckFinalizeComputeParams
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisOnline_cCheckFinalizeComputeParams(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Analysis<online> > alg = staticPointerCast<Analysis<online>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->checkFinalizeComputeParams());
}

/*
 * Class:     com_intel_daal_algorithms_AnalysisOnline
 * Method:    cCheckComputeParams
 * Signature:(J)I
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisOnline_cCheckComputeParams(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    SharedPtr<Analysis<online> > alg = staticPointerCast<Analysis<online>, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)algAddr);
    DAAL_CHECK_THROW(alg->checkComputeParams());
}

/*
 * Class:     com_intel_daal_algorithms_AnalysisOnline
 * Method:    cDispose
 * Signature:(J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_AnalysisOnline_cDispose(JNIEnv * env, jobject thisObj, jlong algAddr)
{
    delete (SharedPtr<AlgorithmIface> *)algAddr;
}

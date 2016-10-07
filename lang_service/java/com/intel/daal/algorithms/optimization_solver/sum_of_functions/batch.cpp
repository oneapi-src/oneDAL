/* file: batch.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
#include "optimization_solver/sum_of_functions/JBatch.h"
#include "common_defines.i"
#include "java_batch.h"

using namespace daal::services;
using namespace daal;
using namespace daal::algorithms::optimization_solver::sum_of_functions;
using namespace daal::algorithms;

extern "C"
{
    /*
     * Class:     com_intel_daal_algorithms_optimization_solver_sum_of_functions_Batch
     * Method:    cDispose
     * Signature: (J)V
     */
    JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sum_1of_1functions_Batch_cDispose
    (JNIEnv *env, jobject thisObj, jlong cBatchIface)
    {
        SharedPtr<AlgorithmIface> *batchIface = (SharedPtr<AlgorithmIface> *)cBatchIface;
        delete batchIface;
    }

    /*
    * Class:     com_intel_daal_algorithms_optimization_solver_sum_of_functions_Batch
    * Method:    cInitBatchIface
    * Signature: (J)J
    */
    JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sum_1of_1functions_Batch_cInitBatchIface
    (JNIEnv *env, jobject thisObj, jlong numberOfTerms)
    {
        JavaVM *jvm;

        // Get pointer to the Java VM interface function table
        jint status = env->GetJavaVM(&jvm);
        if(status != 0)
        {
            env->ThrowNew(env->FindClass("java/lang/Exception"), "Unable to get pointer to the Java VM interface function table");
            return 0;
        }
        SharedPtr<AlgorithmIface> *batchIface = new SharedPtr<AlgorithmIface>(new JavaBatch(numberOfTerms, jvm, thisObj));

        return (jlong)batchIface;
    }

    /*
     * Class:     com_intel_daal_algorithms_optimization_solver_sum_of_functions_Batch
     * Method:    cSetPointersToIface
     * Signature: (JJJ)V
     */
    JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_optimization_1solver_sum_1of_1functions_Batch_cSetPointersToIface
    (JNIEnv *, jobject, jlong cBatchIface, jlong cInput, jlong cParameter)
    {
        SharedPtr<JavaBatch> alg = staticPointerCast<JavaBatch, AlgorithmIface>(*(SharedPtr<AlgorithmIface> *)cBatchIface);

        alg->setPointersToContainer(((sum_of_functions::Input *)cInput), ((sum_of_functions::Parameter *)cParameter));
    }
}

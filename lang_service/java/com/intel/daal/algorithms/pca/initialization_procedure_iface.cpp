/* file: initialization_procedure_iface.cpp */
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
#include "initialization_procedure.h"

#include "pca/JInitializationProcedureIface.h"
#include "pca/JMethod.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::pca;

#define CorrelationDenseValue com_intel_daal_algorithms_pca_Method_correlationDenseValue
#define SVDDenseValue         com_intel_daal_algorithms_pca_Method_svdDenseValue

/*
 * Class:     com_intel_daal_algorithms_pca_InitializationProcedureIface
 * Method:    cNewJavaInitializationProcedure
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL
Java_com_intel_daal_algorithms_pca_InitializationProcedureIface_cNewJavaInitializationProcedure
(JNIEnv *env, jobject thisObj, jint method)
{
    JavaVM *jvm;

    // Get pointer to the Java VM interface function table
    jint status = env->GetJavaVM(&jvm);
    if(status != 0)
    {
        return 0;
    }

    if(method == CorrelationDenseValue)
    {
        pca::PartialResultsInitIface<defaultDense> *initializationProcedure
            = new pca::JavaPartialResultInit<defaultDense>(jvm, thisObj);

        services::SharedPtr<pca::PartialResultsInitIface<defaultDense> > *initializationProcedureShPtr =
            new services::SharedPtr<pca::PartialResultsInitIface<defaultDense> >(initializationProcedure);

        return (jlong)initializationProcedureShPtr;
    }
    else if(method == SVDDenseValue)
    {
        pca::PartialResultsInitIface<svdDense> *initializationProcedure
            = new pca::JavaPartialResultInit<svdDense>(jvm, thisObj);

        services::SharedPtr<pca::PartialResultsInitIface<svdDense> > *initializationProcedureShPtr =
            new services::SharedPtr<pca::PartialResultsInitIface<svdDense> >(initializationProcedure);

        return (jlong)initializationProcedureShPtr;
    }
    return (jlong)0;
}

/*
 * Class:     com_intel_daal_algorithms_pca_InitializationProcedureIface
 * Method:    cDispose
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_InitializationProcedureIface_cDispose
(JNIEnv *env, jobject thisObj, jlong initAddr, jint method)
{
    if(method == CorrelationDenseValue)
    {
        services::SharedPtr<pca::PartialResultsInitIface<defaultDense> > *initializationProcedureShPtr =
            (services::SharedPtr<pca::PartialResultsInitIface<defaultDense> > *)initAddr;
        delete initializationProcedureShPtr;
    }
    else if(method == SVDDenseValue)
    {
        services::SharedPtr<pca::PartialResultsInitIface<svdDense> > *initializationProcedureShPtr =
            (services::SharedPtr<pca::PartialResultsInitIface<svdDense> > *)initAddr;
        delete initializationProcedureShPtr;
    }
}

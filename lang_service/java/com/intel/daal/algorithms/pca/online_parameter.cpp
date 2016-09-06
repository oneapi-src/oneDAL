/* file: online_parameter.cpp */
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
#include "pca/JOnline.h"
#include "pca/JMethod.h"
#include "pca/JOnlineParameter.h"
#include "JComputeMode.h"
#include "JComputeStep.h"

#define CorrelationDenseValue com_intel_daal_algorithms_pca_Method_correlationDenseValue
#define SVDDenseValue         com_intel_daal_algorithms_pca_Method_svdDenseValue

/*
 * Class:     com_intel_daal_algorithms_pca_OnlineParameter
 * Method:    cSetInitializationProcedure
 * Signature: (JJIIII)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_OnlineParameter_cSetInitializationProcedure
(JNIEnv *env, jobject thisObj, jlong parAddr, jlong initAddr, jint method, jint cmode, jint computeStep, jint prec)
{
    using namespace daal::algorithms::pca;
    using namespace daal::services;

    if(method == CorrelationDenseValue)
    {
        if(prec == 0) //double
        {
            OnlineParameter<double, defaultDense> *parameterAddrCor = (OnlineParameter<double, defaultDense> *)parAddr;
            parameterAddrCor->initializationProcedure = *((SharedPtr<PartialResultsInitIface<defaultDense> > *)initAddr);
        }
        else if (prec == 1) //float
        {
            OnlineParameter<float, defaultDense> *parameterAddrCor = (OnlineParameter<float, defaultDense> *)parAddr;
            parameterAddrCor->initializationProcedure = *((SharedPtr<PartialResultsInitIface<defaultDense> > *)initAddr);
        }
    }
}

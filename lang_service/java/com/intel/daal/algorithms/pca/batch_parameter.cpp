/* file: batch_parameter.cpp */
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
#include "com_intel_daal_algorithms_pca_Online.h"
#include "com_intel_daal_algorithms_pca_Method.h"
#include "com_intel_daal_algorithms_pca_BatchParameter.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::services;
using namespace daal::data_management;

#define CorrelationDenseValue com_intel_daal_algorithms_pca_Method_correlationDenseValue
#define SVDDenseValue         com_intel_daal_algorithms_pca_Method_svdDenseValue

#include "common_defines.i"

/*
 * Class:     com_intel_daal_algorithms_pca_BatchParameter
 * Method:    cSetCovariance
 * Signature: (JJJIIII)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_BatchParameter_cSetCovariance(JNIEnv * env, jobject thisObj, jlong parAddr,
                                                                                        jlong covarianceAddr, jint method, jint cmode,
                                                                                        jint computeStep, jint prec)
{
    using namespace daal::algorithms::pca;

    if (method == CorrelationDenseValue)
    {
        if (cmode == jBatch)
        {
            if (prec == 0) //double
            {
                BatchParameter<double, pca::correlationDense> * parameterAddr = (BatchParameter<double, pca::correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::BatchImpl> *)covarianceAddr);
            }
            else
            {
                BatchParameter<float, pca::correlationDense> * parameterAddr = (BatchParameter<float, pca::correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::BatchImpl> *)covarianceAddr);
            }
        }
        else if (cmode == jOnline || (cmode == jDistributed && computeStep == jStep1Local))
        {
            if (prec == 0) //double
            {
                OnlineParameter<double, pca::correlationDense> * parameterAddr = (OnlineParameter<double, pca::correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::OnlineImpl> *)covarianceAddr);
            }
            else
            {
                OnlineParameter<float, pca::correlationDense> * parameterAddr = (OnlineParameter<float, pca::correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::OnlineImpl> *)covarianceAddr);
            }
        }
        else if (cmode == jDistributed)
        {
            if (prec == 0) //double
            {
                DistributedParameter<step2Master, double, pca::correlationDense> * parameterAddr =
                    (DistributedParameter<step2Master, double, pca::correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::DistributedIface<step2Master> > *)covarianceAddr);
            }
            else
            {
                DistributedParameter<step2Master, float, pca::correlationDense> * parameterAddr =
                    (DistributedParameter<step2Master, float, pca::correlationDense> *)parAddr;
                parameterAddr->covariance = *((SharedPtr<daal::algorithms::covariance::DistributedIface<step2Master> > *)covarianceAddr);
            }
        }
    }
}

/*
 * Class:     com_intel_daal_algorithms_pca_BatchParameter
 * Method:    cSetResultsToCompute
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_BatchParameter_cSetResultsToCompute(JNIEnv *, jobject, jlong parAddr,
                                                                                              jlong resultsToCompute, jint method)
{
    if (method == CorrelationDenseValue)
    {
        ((pca::BatchParameter<double, pca::correlationDense> *)parAddr)->resultsToCompute = resultsToCompute;
    }
    else if (method == SVDDenseValue)
    {
        ((pca::BatchParameter<double, pca::svdDense> *)parAddr)->resultsToCompute = resultsToCompute;
    }
}

/*
 * Class:     com_intel_daal_algorithms_pca_BatchParameter
 * Method:    cGetResultsToCompute
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_BatchParameter_cGetResultsToCompute(JNIEnv *, jobject, jlong parAddr, jint method)
{
    return (method == CorrelationDenseValue) ? ((pca::BatchParameter<double, pca::correlationDense> *)parAddr)->resultsToCompute :
                                               ((pca::BatchParameter<double, pca::svdDense> *)parAddr)->resultsToCompute;
}

/*
 * Class:     com_intel_daal_algorithms_pca_BatchParameter
 * Method:    cSetIsDeterministic
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_BatchParameter_cSetIsDeterministic(JNIEnv *, jobject, jlong parAddr,
                                                                                             jboolean isDeterministic, jint method)
{
    if (method == CorrelationDenseValue)
    {
        ((pca::BatchParameter<double, pca::correlationDense> *)parAddr)->isDeterministic = isDeterministic;
    }
    else if (method == SVDDenseValue)
    {
        ((pca::BatchParameter<double, pca::svdDense> *)parAddr)->isDeterministic = isDeterministic;
    }
}

/*
 * Class:     com_intel_daal_algorithms_pca_BatchParameter
 * Method:    cGetIsDeterministic
 * Signature: (J)J
 */
JNIEXPORT jboolean JNICALL Java_com_intel_daal_algorithms_pca_BatchParameter_cGetIsDeterministic(JNIEnv *, jobject, jlong parAddr, jint method)
{
    return (method == CorrelationDenseValue) ? ((pca::BatchParameter<double, pca::correlationDense> *)parAddr)->isDeterministic :
                                               ((pca::BatchParameter<double, pca::svdDense> *)parAddr)->isDeterministic;
}

/*
 * Class:     com_intel_daal_algorithms_pca_BatchParameter
 * Method:    cSetNumberOfcomponents
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_intel_daal_algorithms_pca_BatchParameter_cSetNumberOfComponents(JNIEnv *, jobject, jlong parAddr,
                                                                                                jboolean nComponents, jint method)
{
    if (method == CorrelationDenseValue)
    {
        ((pca::BatchParameter<double, pca::correlationDense> *)parAddr)->nComponents = nComponents;
    }
    else if (method == SVDDenseValue)
    {
        ((pca::BatchParameter<double, pca::svdDense> *)parAddr)->nComponents = nComponents;
    }
}

/*
 * Class:     com_intel_daal_algorithms_pca_BatchParameter
 * Method:    cGetNumberOfComponents
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_intel_daal_algorithms_pca_BatchParameter_cGetNumberOfComponents(JNIEnv *, jobject, jlong parAddr, jint method)
{
    return (method == CorrelationDenseValue) ? ((pca::BatchParameter<double, pca::correlationDense> *)parAddr)->nComponents :
                                               ((pca::BatchParameter<double, pca::svdDense> *)parAddr)->nComponents;
}

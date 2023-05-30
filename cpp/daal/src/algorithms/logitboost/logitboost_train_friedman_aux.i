/* file: logitboost_train_friedman_aux.i */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/*
//++
//  Implementation of auxiliary functions for Logit Boost
//  LBFriedman method.
//--
*/
/*
//
//  REFERENCES
//
//  1. J. Friedman, T. Hastie, R. Tibshirani.
//     Additive logistic regression: a statistical view of boosting,
//     The annals of Statistics, 2000, v28 N2, pp. 337-407
//  2. J. Friedman, T. Hastie, R. Tibshirani.
//     The Elements of Statistical Learning:
//     Data Mining, Inference, and Prediction,
//     Springer, 2001.
//
*/

#ifndef __LOGITBOOST_TRAIN_FRIEDMAN_AUX_I__
#define __LOGITBOOST_TRAIN_FRIEDMAN_AUX_I__

#include "src/services/service_data_utils.h"

using namespace daal::algorithms::logitboost::internal;

namespace daal
{
namespace algorithms
{
namespace logitboost
{
namespace training
{
namespace internal
{
/**
 *  \brief Update working responses and weights for current class
 *
 *  \param n[in]        Number of observations in training data set
 *  \param nc[in]       Number of classes in training data set
 *  \param curClass[in] Index of the current class
 *  \param label[in]    Array of class labels for each observation
 *  \param P[in]        Matrix of probabilities of size nc x n
 *  \param thrW[in]     Threshold for weight calculations
 *  \param w[out]       Array of weights of size n
 *  \param thrZ[in]     Threshold for responses calculations
 *  \param w[out]       Array of responses of size n
 */
template <typename algorithmFPType, CpuType cpu>
void initWZ(size_t n, size_t nc, size_t curClass, const int * label, const algorithmFPType * P, algorithmFPType thrW, algorithmFPType * w,
            algorithmFPType thrZ, algorithmFPType * z)
{
    const algorithmFPType one      = (algorithmFPType)1.0;
    const algorithmFPType three    = (algorithmFPType)3.0;
    const algorithmFPType negThree = (algorithmFPType)(-3.0);
    DAAL_ASSERT(curClass <= services::internal::MaxVal<int>::get())
    int iCurClass                = (int)curClass;
    const algorithmFPType * Pptr = P + curClass * n;

    algorithmFPType sumW = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        algorithmFPType p = Pptr[i];
        w[i]              = p * (one - p);
        if (thrW > w[i])
        {
            w[i] = thrW;
        }
        sumW += w[i];

        if (label[i] == iCurClass)
        {
            if (p > thrZ)
            {
                z[i] = one / p;
            }
            else
            {
                z[i] = three;
            }
        }
        else
        {
            if (one - p > thrZ)
            {
                z[i] = -one / (one - p);
            }
            else
            {
                z[i] = negThree;
            }
        }
    }

    algorithmFPType invSumW = one / sumW;
    for (size_t i = 0; i < n; i++)
    {
        w[i] *= invSumW;
    }
}

/**
 *  \brief Calculate training accuracy
 *
 *  \param n[in]        Number of observations
 *  \param nc[in]       Number of classes
 *  \param y_label[in]  Classes labels
 *  \param P[in]        Array of probabilities of size nc x n
 *  \param lCurPtr[out] Log-likelihood of the model
 *  \param accPtr[out]  Training accuracy
 */
template <typename algorithmFPType, CpuType cpu>
void calculateAccuracy(size_t n, size_t nc, const int * y_label, algorithmFPType * P, algorithmFPType & lCur, algorithmFPType & acc)
{
    algorithmFPType lPrev = lCur;
    lCur                  = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        lCur -= daal::internal::MathInst<algorithmFPType, cpu>::sLog(P[y_label[i] * n + i]);
    }
    const algorithmFPType diff = daal::internal::MathInst<algorithmFPType, cpu>::sFabs(lPrev - lCur);
    acc                        = daal::internal::MathInst<algorithmFPType, cpu>::sMin(diff, diff / (lPrev + (algorithmFPType)1e-6));
}

} // namespace internal
} // namespace training
} // namespace logitboost
} // namespace algorithms
} // namespace daal

#endif

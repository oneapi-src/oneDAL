/* file: stump_train_impl.i */
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

/*
//++
//  Implementation of Fast method for Decision Stump algorithm.
//--
*/

#ifndef __STUMP_TRAIN_IMPL_I__
#define __STUMP_TRAIN_IMPL_I__

#include "threading.h"
#include "numeric_table.h"
#include "service_utils.h"
#include "service_data_utils.h"
#include "service_memory.h"
#include "stump_train_kernel.h"
#include "service_numeric_table.h"
#include "service_error_handling.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace training
{
namespace internal
{
using namespace daal::internal;

/**
 *  \brief Quick sort function that sorts array x and rearranges arrays w,z
 *         accordingly
 *
 *  \param n[in] Length of input arrays
 *  \param x     Array that is used as "key" when sorted
 *  \param w     Array that is used as "value" when sorted
 *  \param z     Array that is used as "value" when sorted
 */
template <Method method, typename algorithmFPtype, CpuType cpu>
void StumpTrainKernel<method, algorithmFPtype, cpu>::StumpQSort(size_t n, algorithmFPtype * x, algorithmFPtype * w, algorithmFPtype * z)
{
    int i, ir, j, k, jstack = -1, l = 0;
    algorithmFPtype a, b, c, tmp;
    const int M = 7, NSTACK = 128;
    algorithmFPtype istack[NSTACK];

    ir = n - 1;

    for (;;)
    {
        if (ir - l < M)
        {
            for (j = l + 1; j <= ir; j++)
            {
                a = x[j];
                b = w[j];
                c = z[j];

                for (i = j - 1; i >= l; i--)
                {
                    if (x[i] <= a)
                    {
                        break;
                    }
                    x[i + 1] = x[i];
                    w[i + 1] = w[i];
                    z[i + 1] = z[i];
                }

                x[i + 1] = a;
                w[i + 1] = b;
                z[i + 1] = c;
            }

            if (jstack < 0)
            {
                break;
            }

            ir = istack[jstack--];
            l  = istack[jstack--];
        }
        else
        {
            k = (l + ir) >> 1;
            daal::services::internal::swap<cpu, algorithmFPtype>(x[k], x[l + 1]);
            daal::services::internal::swap<cpu, algorithmFPtype>(w[k], w[l + 1]);
            daal::services::internal::swap<cpu, algorithmFPtype>(z[k], z[l + 1]);
            if (x[l] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmFPtype>(x[l], x[ir]);
                daal::services::internal::swap<cpu, algorithmFPtype>(w[l], w[ir]);
                daal::services::internal::swap<cpu, algorithmFPtype>(z[l], z[ir]);
            }
            if (x[l + 1] > x[ir])
            {
                daal::services::internal::swap<cpu, algorithmFPtype>(x[l + 1], x[ir]);
                daal::services::internal::swap<cpu, algorithmFPtype>(w[l + 1], w[ir]);
                daal::services::internal::swap<cpu, algorithmFPtype>(z[l + 1], z[ir]);
            }
            if (x[l] > x[l + 1])
            {
                daal::services::internal::swap<cpu, algorithmFPtype>(x[l], x[l + 1]);
                daal::services::internal::swap<cpu, algorithmFPtype>(w[l], w[l + 1]);
                daal::services::internal::swap<cpu, algorithmFPtype>(z[l], z[l + 1]);
            }
            i = l + 1;
            j = ir;
            a = x[l + 1];
            b = w[l + 1];
            c = z[l + 1];
            for (;;)
            {
                while (x[++i] < a)
                    ;
                while (x[--j] > a)
                    ;
                if (j < i)
                {
                    break;
                }
                daal::services::internal::swap<cpu, algorithmFPtype>(x[i], x[j]);
                daal::services::internal::swap<cpu, algorithmFPtype>(w[i], w[j]);
                daal::services::internal::swap<cpu, algorithmFPtype>(z[i], z[j]);
            }
            x[l + 1] = x[j];
            w[l + 1] = w[j];
            z[l + 1] = z[j];

            x[j] = a;
            w[j] = b;
            z[j] = c;
            jstack += 2;

            if (ir - i + 1 >= j - l)
            {
                istack[jstack]     = ir;
                istack[jstack - 1] = i;
                ir                 = j - 1;
            }
            else
            {
                istack[jstack]     = j - 1;
                istack[jstack - 1] = l;
                l                  = i;
            }
        }
    }

    return;
}

/**
 *  \brief Fit the function f[j] by a weighted least-squares
 *  regression of x to z with weigths w.
 *  Process ordered or numerical feature
 *
 *  \param n[in]        Number of observations
 *  \param x[in]        Input data feature of size n
 *  \param w[in]        Array of weights of size n
 *  \param z[in]        Array of weights of responses of size n
 *  \param sumW[in]     Total sum of weights
 *  \param sumM[in]     Total sum of weighted responses
 *  \param sumS[in]     Total sum of weighted squares of responses
 *  \param minS[out]       Value of goal function obtained for the best split
 *  \param splitPoint[out] Resulting split point
 *  \param lMean[out]      "left" average of weighted responses
 *                            for resulting split
 *  \param rMean[out]      "right" average of weighted responses
 *                            for resulting split
 */
template <Method method, typename algorithmFPtype, CpuType cpu>
services::Status StumpTrainKernel<method, algorithmFPtype, cpu>::stumpRegressionOrdered(
    size_t nVectors, const algorithmFPtype * x, const algorithmFPtype * w, const algorithmFPtype * z, algorithmFPtype sumW, algorithmFPtype sumM,
    algorithmFPtype sumS, algorithmFPtype & minS, algorithmFPtype & splitPoint, algorithmFPtype & lMean, algorithmFPtype & rMean)
{
    const algorithmFPtype THR = 1e-10;
    const algorithmFPtype C05 = (algorithmFPtype)0.5;
    splitPoint                = 0.0;
    lMean                     = 0.0;
    rMean                     = 0.0;
    int result                = 0;

    algorithmFPtype lw, rw; /* sums of weights in the left and right regions */
    algorithmFPtype lm, rm; /* weighted means of the responses z of the left and right
                       regions (denoted as c1 and c2 in part 9.2.2 in [2]) */
    algorithmFPtype lM, rM; /* weighted sums of the responses z of the left and right regions */
    algorithmFPtype ls, rs; /* weighted sum of squares of the responses z */
    algorithmFPtype sum;    /* goal function (minimization criteria, see (9.13) in [2]) */
    algorithmFPtype lc, rc; /* goal functions of the left and right regions
                      (see (9.13) in [2]) */

    /* Allocate memory for storing intermediate data */
    TArray<algorithmFPtype, cpu> aXX(nVectors);
    TArray<algorithmFPtype, cpu> aWW(nVectors);
    TArray<algorithmFPtype, cpu> aZZ(nVectors);
    algorithmFPtype * xx = aXX.get();
    algorithmFPtype * ww = aWW.get();
    algorithmFPtype * zz = aZZ.get();

    DAAL_CHECK(xx && ww && zz, services::ErrorMemoryAllocationFailed);

    result |= daal::services::internal::daal_memcpy_s(ww, nVectors * sizeof(algorithmFPtype), w, nVectors * sizeof(algorithmFPtype));
    result |= daal::services::internal::daal_memcpy_s(zz, nVectors * sizeof(algorithmFPtype), z, nVectors * sizeof(algorithmFPtype));
    result |= daal::services::internal::daal_memcpy_s(xx, nVectors * sizeof(algorithmFPtype), x, nVectors * sizeof(algorithmFPtype));
    DAAL_CHECK(!result, services::ErrorMemoryCopyFailedInternal);

    StumpQSort(nVectors, xx, ww, zz);

    lw = 0.0;
    lM = 0.0;
    ls = 0.0;
    rw = sumW;
    rM = sumM;
    rs = sumS;

    /* Seek split point s. */
    for (size_t k = 0; k < nVectors - 1; k++)
    {
        /* Move points one-by-one from the right regoin into the left
           and choose the optimal split */
        algorithmFPtype wz  = ww[k] * zz[k];
        algorithmFPtype wzz = wz * zz[k];

        lw += ww[k];
        lM += wz;
        ls += wzz;
        rw -= ww[k];
        rM -= wz;
        rs -= wzz;

        if (xx[k] == xx[k + 1])
        {
            continue;
        }

        /* Current split point */
        algorithmFPtype curT = C05 * (xx[k] + xx[k + 1]);

        /* Calculate weight; weighted mean and weighted sum of squares
           over points left to curT */

        /* Calculate goal function (lc = Sum (ww[j]*(zz[j] - lm)*(zz[j] - lm))
           for the left region
           (See left term of (9.13) in [2]) */
        lm = 0.0;
        lc = 0.0;
        if (lw > THR)
        {
            /* Calculate the optimal solution for the left region
               (See (9.11) in [2])*/
            lm = lM / lw;
            lc = ls - lM * lm;
        }

        /* Calculate weight; weighted mean and weighted sum of squares
           over points right to curT */

        /* Calculate goal function (rc = Sum (ww[j]*(zz[j] - rm)*(zz[j] - rm))
           for the right region
           (See left right of (9.13) in [2]) */
        rm = 0.0;
        rc = 0.0;
        if (rw > THR)
        {
            /* Calculate the optimal solution for the right region
               (See (9.11) in [2])*/
            rm = rM / rw;
            rc = rs - rM * rm;
        }
        /* Calculate goal function for the current split (See (9.13) in [2]) */
        sum = lc + rc;

        if (sum < minS)
        {
            /* remember the minimal split point and weighted means */
            minS       = sum;
            splitPoint = curT;
            lMean      = lm;
            rMean      = rm;
        }
    }
    return services::Status();
}

/**
 *  \brief Fit the function f[m*nk+j] by a weighted least-squares
 *  regression of z to x with weigths w.
 *  Process categorical feature
 *
 *  \param n[in]              Number of observations
 *  \param nCategories[in]    Number of categories in input feature x
 *  \param x[in]              Input dataset feature
 *  \param w[in]              Array of weights of size n
 *  \param z[in]              Array of weights of responses of size n
 *  \param sumW[in]           Total sum of weights
 *  \param sumM[in]           Total sum of weighted responses
 *  \param sumS[in]           Total sum of weighted squares of responses
 *  \param minS[out]          Value of goal function obtained for the best split
 *  \param splitPoint[out]    Resulting split point
 *  \param lMean[out]         "left" average of weighted responses
 *                            for resulting split
 *  \param rMean[out]         "right" average of weighted responses
 *                            for resulting split
 */
template <Method method, typename algorithmFPtype, CpuType cpu>
services::Status StumpTrainKernel<method, algorithmFPtype, cpu>::stumpRegressionCategorical(
    size_t n, size_t nCategories, const int * x, const algorithmFPtype * w, const algorithmFPtype * z, algorithmFPtype sumW, algorithmFPtype sumM,
    algorithmFPtype sumS, algorithmFPtype & minS, algorithmFPtype & splitPoint, algorithmFPtype & lMean, algorithmFPtype & rMean)
{
    DAAL_ASSERT(nCategories >= 2);

    TArray<algorithmFPtype, cpu> aW_per_cat(nCategories);
    TArray<algorithmFPtype, cpu> aM_per_cat(nCategories);
    TArray<algorithmFPtype, cpu> aS_per_cat(nCategories);

    algorithmFPtype * W_per_cat = aW_per_cat.get();
    algorithmFPtype * M_per_cat = aM_per_cat.get();
    algorithmFPtype * S_per_cat = aS_per_cat.get();

    DAAL_CHECK(W_per_cat && M_per_cat && S_per_cat, services::ErrorMemoryAllocationFailed);

    const algorithmFPtype zero = 0.0;
    for (size_t i = 0; i < nCategories; i++)
    {
        W_per_cat[i] = zero;
        M_per_cat[i] = zero;
        S_per_cat[i] = zero;
    }

    /* Calculate weight; weighted mean and weighted sum of squares
       for each category */
    for (size_t i = 0; i < n; i++)
    {
        int iCat           = x[i];
        algorithmFPtype wz = w[i] * z[i];
        W_per_cat[iCat] += w[i];
        M_per_cat[iCat] += wz;
        S_per_cat[iCat] += wz * z[i];
    }

    const algorithmFPtype THR = 1e-10;
    splitPoint                = 0.0;
    lMean                     = 0.0;
    rMean                     = 0.0;

    algorithmFPtype curT;
    algorithmFPtype lm, rm; /* weighted means of the responses z of the left and right
                       regions (denoted as c1 and c2 in part 9.2.2 in [2]) */
    algorithmFPtype sum;    /* goal function (minimization criteria, see (9.13) in [2]) */
    algorithmFPtype lc, rc; /* goal functions of the left and right regions
                      (see (9.13) in [2]) */
    /* Use greedy algorithm to find an optimal split:
       Find the split that minimizes the goal function (lc + rc)
       among the splits that have 1 category in the left regoin
       and the rest nCategories-1 categories in the right region.

       This is a heuristics, because checking total number of possible
       splits (which is pow(2,nCategories)-1) could be time consuming */
    for (size_t i = 0; i < nCategories; i++)
    {
        // x[i] - current split point
        curT               = (algorithmFPtype)x[i];
        algorithmFPtype lM = M_per_cat[i]; /* weighted sums of the responses z of the left and right regions */
        algorithmFPtype rM = sumM - lM;
        algorithmFPtype lw = W_per_cat[i]; /* sums of weights in the left and right regions */
        algorithmFPtype rw = sumW - lw;
        algorithmFPtype ls = S_per_cat[i]; /* weighted sum of squares of the responses z */
        algorithmFPtype rs = sumS - ls;    /* weighted sum of squares of the responses z */

        /* Calculate weight; weighted mean and weighted sum of squares
           over points left to curT */

        /* Calculate goal function (lc = Sum (ww[j]*(zz[j] - lm)*(zz[j] - lm))
           for the left region
           (See left term of (9.13) in [2]) */
        lm = 0.0;
        lc = 0.0;
        if (lw > THR)
        {
            /* Calculate the optimal solution for the left region
               (See (9.11) in [2])*/
            lm = lM / lw;
            lc = ls - lM * lm;
        }

        /* Calculate weight; weighted mean and weighted sum of squares
           over points right to curT */

        /* Calculate goal function (rc = Sum (ww[j]*(zz[j] - rm)*(zz[j] - rm))
           for the right region
           (See left right of (9.13) in [2]) */
        rm = 0.0;
        rc = 0.0;
        if (rw > THR)
        {
            /* Calculate the optimal solution for the right region
               (See (9.11) in [2])*/
            rm = rM / rw;
            rc = rs - rM * rm;
        }
        /* Calculate goal function for the current split (See (9.13) in [2]) */
        sum = lc + rc;

        if (sum < minS)
        {
            /* remember the minimal split point and weighted means */
            minS       = sum;
            splitPoint = curT;
            lMean      = lm;
            rMean      = rm;
        }
    }
    return services::Status();
}

/**
 *  \brief Calculate total sum of weights, weighted responses and
 *         weighted squares of responses
 *
 *  \param n[in]        Number of observations in training data set
 *  \param w[in]        Array of weights of size n
 *  \param z[in]        Array of responses of size n
 *  \param sumW[out]    Total sum of weights
 *  \param sumM[out]    Total sum of weighted responses
 *  \param sumS[out]    Total sum of weighted squares of responses
 */
template <Method method, typename algorithmFPtype, CpuType cpu>
void StumpTrainKernel<method, algorithmFPtype, cpu>::computeSums(size_t n, const algorithmFPtype * w, const algorithmFPtype * z,
                                                                 algorithmFPtype & sumW, algorithmFPtype & sumM, algorithmFPtype & sumS)
{
    sumW = sumM = sumS = (algorithmFPtype)0.0;
    for (size_t i = 0; i < n; i++)
    {
        algorithmFPtype wz = w[i] * z[i];
        sumW += w[i];
        sumM += wz;
        sumS += wz * z[i];
    }
}

template <typename algorithmFPtype, CpuType cpu>
struct group_res
{
    DAAL_NEW_DELETE();
    size_t groupSplitFeature;
    algorithmFPtype groupSplitPoint;
    algorithmFPtype groupLMean;
    algorithmFPtype groupRMean;
    algorithmFPtype groupMinS;
};

template <Method method, typename algorithmFPtype, CpuType cpu>
services::Status StumpTrainKernel<method, algorithmFPtype, cpu>::doStumpRegression(size_t n, size_t dim, const NumericTable * x,
                                                                                   const algorithmFPtype * w, const algorithmFPtype * z,
                                                                                   size_t & splitFeature, algorithmFPtype & splitPoint,
                                                                                   algorithmFPtype & leftValue, algorithmFPtype & rightValue)
{
    algorithmFPtype minS = daal::services::internal::MaxVal<algorithmFPtype>::get();
    algorithmFPtype sumW, sumM, sumS;
    computeSums(n, w, z, sumW, sumM, sumS);
    typedef group_res<algorithmFPtype, cpu> TGroupRes;
    daal::tls<TGroupRes *> tls([=]() -> TGroupRes * {
        TGroupRes * g = new TGroupRes();
        DAAL_CHECK_STATUS_VAR(g)
        g->groupMinS = daal::services::internal::MaxVal<algorithmFPtype>::get();
        return g;
    });

    SafeStatus safeStat;
    daal::services::AtomicInt nCategoricalFeaturesSkipped(0);
    daal::threader_for(dim, dim, [=, &safeStat, &nCategoricalFeaturesSkipped, &tls](size_t k) {
        algorithmFPtype localSplitPoint;
        algorithmFPtype localLMean;
        algorithmFPtype localRMean;
        algorithmFPtype localMinS = daal::services::internal::MaxVal<algorithmFPtype>::get();

        services::Status s;
        if (x->getFeatureType(k) == data_management::features::DAAL_CATEGORICAL)
        {
            /* Here if feature k is categorical */
            const size_t nCategories = x->getNumberOfCategories(k);
            if (nCategories < 2)
            {
                nCategoricalFeaturesSkipped.inc();
                return;
            }
            ReadColumns<int, cpu> block(*const_cast<NumericTable *>(x), k, (size_t)0, n);
            s = block.status();
            if (s)
                s = stumpRegressionCategorical(n, nCategories, block.get(), w, z, sumW, sumM, sumS, localMinS, localSplitPoint, localLMean,
                                               localRMean);
        }
        else
        {
            /* Here if feature k is not categorical */
            ReadColumns<algorithmFPtype, cpu> block(*const_cast<NumericTable *>(x), k, (size_t)0, n);
            s = block.status();
            if (s) s = stumpRegressionOrdered(n, block.get(), w, z, sumW, sumM, sumS, localMinS, localSplitPoint, localLMean, localRMean);
        }
        if (!s)
        {
            safeStat |= s;
            return;
        }

        TGroupRes * local = tls.local();
        DAAL_CHECK_THR(local, ErrorMemoryAllocationFailed);
        if (localMinS < local->groupMinS)
        {
            local->groupMinS         = localMinS;
            local->groupSplitFeature = k;
            local->groupSplitPoint   = localSplitPoint;
            local->groupLMean        = localLMean;
            local->groupRMean        = localRMean;
        }
    });

    tls.reduce([&](TGroupRes * g) {
        if (safeStat && (g->groupMinS < minS))
        {
            minS         = g->groupMinS;
            splitFeature = g->groupSplitFeature;
            splitPoint   = (algorithmFPtype)(g->groupSplitPoint);
            leftValue    = (algorithmFPtype)(g->groupLMean);
            rightValue   = (algorithmFPtype)(g->groupRMean);
        }

        delete (g);
    });

    if (!safeStat) return safeStat.detach();
    return (int(dim) == nCategoricalFeaturesSkipped.get() ? services::Status(services::ErrorStumpInvalidInputCategoricalData) : services::Status());
}

} // namespace internal
} // namespace training
} // namespace stump
} // namespace algorithms
} // namespace daal

#endif

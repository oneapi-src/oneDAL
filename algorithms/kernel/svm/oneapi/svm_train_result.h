/* file: svm_train_cache.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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
//  SVM cache structure implementation
//--
*/

#ifndef __SVM_TRAIN_WORKSET_H__
#define __SVM_TRAIN_WORKSET_H__

#include "service/kernel/service_utils.h"
#include "algorithms/kernel/svm/oneapi/svm_helper.h"

using namespace daal::services::internal;

namespace daal
{
namespace algorithms
{
namespace svm
{
namespace training
{
namespace internal
{
template <typename algorithmFPType>
class Result
{
public:
    Result(const services::Buffer<algorithmFPType> & alphaBuff, const services::Buffer<algorithmFPType> & fBuff,
           const services::Buffer<algorithmFPType> & yBuff, const algorithmFPType C, const size_t nVectors)
        : _yBuff(yBuff), _alphaBuff(alphaBuff), _fBuff(fBuff), _C(C), _nVectors(nVectors)
    {}

    services::Status setResultsToModel(const NumericTable & xTable, Model & model) const
    {
        {
            const algorithmFPType * alpha = _alphaBuff.toHost(ReadWriteMode::readOnly).get();

            const algorithmFPType zero(0.0);
            size_t nSV = 0;
            for (size_t i = 0; i < _nVectors; i++)
            {
                if (alpha[i] > zero) nSV++;
            }
            printf("nSV %lu\n", nSV);
        }

        model.setNFeatures(xTable.getNumberOfColumns());
        Status s;
        DAAL_CHECK_STATUS(s, setSVCoefficients(nSV, model));
        DAAL_CHECK_STATUS(s, setSVIndices(nSV, model));

        DAAL_CHECK_STATUS(s, setSV_Dense(model, xTable, nSV));

        /* Calculate bias and write it into model */
        model.setBias(double(calculateBias(C)));
        return s;
    }

    services::Status setSVCoefficients(size_t nSV, Model & model) const
    {
        const algorithmFPType zero(0.0);
        NumericTablePtr svCoeffTable = model.getClassificationCoefficients();
        Status s;
        DAAL_CHECK_STATUS(s, svCoeffTable->resize(nSV));

        BlockDescriptor<algorithmFPType> svCoeffBlock;
        DAAL_CHECK_STATUS(status, svCoeffTable->getBlockOfRows(0, nSV, ReadWriteMode::writeOnly, svCoeffBlock));

        algorithmFPType * svCoeff     = svCoeffBlock.getBlockSharedPtr().get();
        const algorithmFPType * y     = _yBuff.toHost(ReadWriteMode::readOnly).get();
        const algorithmFPType * alpha = _alphaBuff.toHost(ReadWriteMode::readOnly).get();

        for (size_t i = 0, iSV = 0; i < _nVectors; i++)
        {
            if (alpha[i] != zero)
            {
                svCoeff[iSV] = y[i] * alpha[i];
                iSV++;
            }
        }
        return s;
    }

    /**
     * \brief Write indices of the support vectors into resulting model
     *
     * \param[in]  nSV          Number of support vectors
     * \param[out] model        Resulting model
     */
    services::Status setSVIndices(size_t nSV, Model & model) const
    {
        NumericTablePtr svIndicesTable = model.getSupportIndices();
        services::Status s;
        DAAL_CHECK_STATUS(s, svIndicesTable->resize(nSV));

        BlockDescriptor<int> svIndicesBlock;
        DAAL_CHECK_STATUS(status, svIndicesTable->getBlockOfRows(0, nSV, ReadWriteMode::writeOnly, svIndicesBlock));

        int * svIndices = svIndicesBlock.getBlockSharedPtr().get();

        const algorithmFPType * alpha = _alphaBuff.toHost(ReadWriteMode::readOnly).get();

        const algorithmFPType zero(0.0);
        for (size_t i = 0, iSV = 0; i < _nVectors; i++)
        {
            if (alpha[i] != zero)
            {
                DAAL_ASSERT(i <= services::internal::MaxVal<int>::get())
                svIndices[iSV++] = int(i);
            }
        }
        return s;
    }

    /**
     * \brief Write support vectors in dense format into resulting model
     *
     * \param[out] model        Resulting model
     * \param[in]  xTable       Input data set in dense layout
     * \param[in]  nSV          Number of support vectors
     */
    Status setSV_Dense(Model & model, const NumericTable & xTable, size_t nSV) const
    {
        const size_t nFeatures = xTable.getNumberOfColumns();
        /* Allocate memory for support vectors and coefficients */
        NumericTablePtr svTable = model.getSupportVectors();
        Status s;
        DAAL_CHECK_STATUS(s, svTable->resize(nSV));
        if (nSV == 0) return s;

        BlockDescriptor<algorithmFPType> svBlock;
        DAAL_CHECK_STATUS(status, svCoeffTable->getBlockOfRows(0, nSV, ReadWriteMode::writeOnly, svBlock));

        auto shSV = svBlock.getBlockSharedPtr();

        algorithmFPType * sv = shSV.get();

        const algorithmFPType zero(0.0);

        BlockDescriptor<algorithmFPType> xBD;
        DAAL_CHECK_STATUS(status, xTable->getBlockOfRows(0, _nVectors, ReadWriteMode::readOnly, xBD));

        auto shX = xBD.getBlockSharedPtr();

        algorithmFPType * x = shX.get();

        const algorithmFPType * alpha = _alphaBuff.toHost(ReadWriteMode::readOnly).get();

        for (size_t i = 0, iSV = 0; i < _nVectors; i++)
        {
            if (alpha[i] == zero) continue;
            const size_t rowIndex      = i;
            const algorithmFPType * xi = x[i];
            for (size_t j = 0; j < nFeatures; j++)
            {
                sv[iSV * nFeatures + j] = xi[j];
            }
            iSV++;
        }
        return s;
    }

    /**
 * \brief Calculate SVM model bias
 *
 * \param[in]  C        Upper bound in constraints of the quadratic optimization problem
 * \return Bias for the SVM model
 */
    algorithmFPType calculateBias(const algorithmFPType C) const
    {
        algorithmFPType bias;
        const algorithmFPType zero(0.0);
        const algorithmFPType one(1.0);
        size_t num_yg          = 0;
        algorithmFPType sum_yg = 0.0;

        const algorithmFPType fpMax = MaxVal<algorithmFPType>::get();
        algorithmFPType ub          = -(fpMax);
        algorithmFPType lb          = fpMax;

        const algorithmFPType * alpha = _alphaBuff.toHost(ReadWriteMode::readOnly).get();
        const algorithmFPType * y     = _alphaBuff.toHost(ReadWriteMode::readOnly).get();
        const algorithmFPType * grad  = _alphaBuff.toHost(ReadWriteMode::readOnly).get();

        for (size_t i = 0; i < _nVectors; i++)
        {
            const algorithmFPType yg = -y[i] * grad[i];
            if (y[i] == -one && alpha[i] == C)
            {
                ub = ((ub > yg) ? ub : yg);
            } /// SVM_MAX(ub, yg);
            else if (y[i] == one && alpha[i] == C)
            {
                lb = ((lb < yg) ? lb : yg);
            } /// SVM_MIN(lb, yg);
            else if (y[i] == one && alpha[i] == zero)
            {
                ub = ((ub > yg) ? ub : yg);
            } /// SVM_MAX(ub, yg);
            else if (y[i] == -one && alpha[i] == zero)
            {
                lb = ((lb < yg) ? lb : yg);
            } /// SVM_MIN(lb, yg);
            else
            {
                sum_yg += yg;
                num_yg++;
            }
        }

        if (num_yg == 0)
        {
            bias = 0.5 * (ub + lb);
        }
        else
        {
            bias = sum_yg / (algorithmFPType)num_yg;
        }

        return bias;
    }

private:
    const services::Buffer<algorithmFPType> _yBuff;
    const services::Buffer<algorithmFPType> _fBuff;
    const services::Buffer<algorithmFPType> _alphaBuff;
    const algorithmFPType _C;
    const size_t _nVectors;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

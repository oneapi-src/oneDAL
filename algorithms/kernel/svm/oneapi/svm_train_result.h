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

#ifndef __SVM_TRAIN_RESULT_H__
#define __SVM_TRAIN_RESULT_H__

#include "service/kernel/service_utils.h"
#include "algorithms/kernel/svm/oneapi/svm_helper.h"
#include "service/kernel/oneapi/sum_reducer.h"

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
class SaveResultModel
{
    using Helper = HelperSVM<algorithmFPType>;

public:
    SaveResultModel(const services::Buffer<algorithmFPType> & alphaBuff, const services::Buffer<algorithmFPType> & fBuff,
                    const services::Buffer<algorithmFPType> & yBuff, const algorithmFPType C, const size_t nVectors)
        : _yBuff(yBuff), _alphaBuff(alphaBuff), _fBuff(fBuff), _C(C), _nVectors(nVectors)
    {}

    services::Status init()
    {
        Status status;
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        _tmpValues     = context.allocate(TypeIds::id<algorithmFPType>(), _nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);
        _mask = context.allocate(TypeIds::id<int>(), _nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);
        return status;
    }

    services::Status setResultsToModel(const NumericTablePtr & xTable, Model & model) const
    {
        Status status;
        const algorithmFPType zero(0.0);
        size_t nSV = 0;
        {
            const algorithmFPType * alpha = _alphaBuff.toHost(ReadWriteMode::readOnly).get();

            for (size_t i = 0; i < _nVectors; i++)
            {
                if (alpha[i] > zero) nSV++;
            }
            printf("nSV %lu\n", nSV);
        }

        model.setNFeatures(xTable->getNumberOfColumns());

        DAAL_CHECK_STATUS(status, setSVCoefficients(nSV, model));
        DAAL_CHECK_STATUS(status, setSVIndices(nSV, model));

        DAAL_CHECK_STATUS(status, setSVDense(model, xTable, nSV));

        /* Calculate bias and write it into model */
        algorithmFPType bias;
        DAAL_CHECK_STATUS(status, calculateBias(_C, bias));
        model.setBias(double(bias));
        return status;
    }

protected:
    services::Status setSVCoefficients(size_t nSV, Model & model) const
    {
        const algorithmFPType zero(0.0);
        NumericTablePtr svCoeffTable = model.getClassificationCoefficients();
        Status status;
        DAAL_CHECK_STATUS(status, svCoeffTable->resize(nSV));

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
        return status;
    }

    services::Status setSVIndices(size_t nSV, Model & model) const
    {
        NumericTablePtr svIndicesTable = model.getSupportIndices();
        services::Status status;
        DAAL_CHECK_STATUS(status, svIndicesTable->resize(nSV));

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
        return status;
    }

    Status setSVDense(Model & model, const NumericTablePtr & xTable, size_t nSV) const
    {
        Status status;

        const size_t nFeatures = xTable->getNumberOfColumns();

        NumericTablePtr svTable = model.getSupportVectors();
        DAAL_CHECK_STATUS(status, svTable->resize(nSV));
        if (nSV == 0) return status;

        BlockDescriptor<algorithmFPType> svBlock;
        DAAL_CHECK_STATUS(status, svTable->getBlockOfRows(0, nSV, ReadWriteMode::writeOnly, svBlock));
        auto svBuff = svBlock.getBuffer();

        NumericTablePtr svIndicesTable = model.getSupportIndices();
        BlockDescriptor<int> svIndicesBlock;
        DAAL_CHECK_STATUS(status, svIndicesTable->getBlockOfRows(0, nSV, ReadWriteMode::readOnly, svIndicesBlock));
        auto svIndicesBuff = svIndicesBlock.getBuffer();

        BlockDescriptor<algorithmFPType> xBlock;
        DAAL_CHECK_STATUS(status, xTable->getBlockOfRows(0, _nVectors, ReadWriteMode::readOnly, xBlock));
        auto xBuff = xBlock.getBuffer();

        DAAL_CHECK_STATUS(status, Helper::copyBlockIndices(xBuff, svIndicesBuff, svBuff, nSV, nFeatures));

        return status;
    }

    Status calculateBias(const algorithmFPType C, algorithmFPType & bias) const
    {
        Status status;
        const algorithmFPType zero(0.0);
        const algorithmFPType one(1.0);
        size_t num_yg          = 0;
        algorithmFPType sum_yg = 0.0;

        const algorithmFPType fpMax = MaxVal<algorithmFPType>::get();
        algorithmFPType ub          = -(fpMax);
        algorithmFPType lb          = fpMax;

        auto tmpValuesBuff = _tmpValues.get<algorithmFPType>();
        auto maskBuff      = _mask.get<int>();

        /* free SV: (0 < alpha < C)*/
        DAAL_CHECK_STATUS(status, Helper::checkFree(_alphaBuff, maskBuff, C, _nVectors));
        size_t nFree = 0;
        DAAL_CHECK_STATUS(status, Helper::gatherValues(maskBuff, _fBuff, _nVectors, tmpValuesBuff, nFree));

        if (nFree > 0)
        {
            auto sumResult = math::SumReducer::sum(math::Layout::RowMajor, tmpValuesBuff, 1, nFree, &status);
            DAAL_CHECK_STATUS_VAR(status);
            auto sumHost = sumResult.sum.get<algorithmFPType>().toHost(data_management::readOnly, &status);
            bias         = -*sumHost / algorithmFPType(nFree);
        }
        else
        {
            algorithmFPType ub = algorithmFPType(0);
            algorithmFPType lb = algorithmFPType(0);
            {
                DAAL_CHECK_STATUS(status, Helper::checkUpper(_yBuff, _alphaBuff, maskBuff, C, _nVectors));
                size_t nUpper = 0;
                DAAL_CHECK_STATUS(status, Helper::gatherValues(maskBuff, _fBuff, _nVectors, tmpValuesBuff, nUpper));
                auto result = math::Reducer::reduce(math::Reducer::BinaryOp::MIN, math::Layout::RowMajor, tmpValuesBuff, 1, nUpper, &status).reduce;
                DAAL_CHECK_STATUS_VAR(status);
                auto minHost = result.get<algorithmFPType>().toHost(data_management::readOnly, &status);
                ub           = *minHost;
            }
            {
                DAAL_CHECK_STATUS(status, Helper::checkLower(_yBuff, _alphaBuff, maskBuff, C, _nVectors));
                size_t nLower = 0;
                DAAL_CHECK_STATUS(status, Helper::gatherValues(maskBuff, _fBuff, _nVectors, tmpValuesBuff, nLower));
                auto result = math::Reducer::reduce(math::Reducer::BinaryOp::MAX, math::Layout::RowMajor, tmpValuesBuff, 1, nLower, &status).reduce;
                DAAL_CHECK_STATUS_VAR(status);
                auto maxHost = result.get<algorithmFPType>().toHost(data_management::readOnly, &status);
                ub           = *maxHost;
            }

            bias = -0.5 * (ub + lb);
        }

        return status;
    }

private:
    const services::Buffer<algorithmFPType> _yBuff;
    const services::Buffer<algorithmFPType> _fBuff;
    const services::Buffer<algorithmFPType> _alphaBuff;
    UniversalBuffer _tmpValues;
    UniversalBuffer _mask;
    const algorithmFPType _C;
    const size_t _nVectors;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

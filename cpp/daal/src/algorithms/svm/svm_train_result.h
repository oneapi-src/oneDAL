/* file: svm_train_result.h */
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
//  SVM save result structure implementation
//--
*/

#ifndef __SVM_TRAIN_RESULT_H__
#define __SVM_TRAIN_RESULT_H__

#include "services/daal_defines.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_memory.h"
#include "src/data_management/service_micro_table.h"
#include "src/data_management/service_numeric_table.h"
#include "src/services/service_utils.h"
#include "src/services/service_data_utils.h"

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
using namespace daal::internal;

/**
* \brief Write support vectors and classification coefficients into output model
*/
template <typename algorithmFPType, CpuType cpu>
class SaveResultTask
{
public:
    SaveResultTask(const size_t nVectors, const algorithmFPType * y, const algorithmFPType * alpha, const algorithmFPType * grad,
                   SVMCacheCommonIface<algorithmFPType, cpu> * cache)
        : _nVectors(nVectors), _y(y), _alpha(alpha), _grad(grad), _cache(cache)
    {}

    services::Status compute(const NumericTable & xTable, Model & model, const algorithmFPType * cw) const
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(saveResult);

        services::Status s;

        const algorithmFPType zero(0.0);
        size_t nSV = 0;
        for (size_t i = 0; i < _nVectors; ++i)
        {
            if (_alpha[i] > zero) nSV++;
        }

        model.setNFeatures(xTable.getNumberOfColumns());
        DAAL_CHECK_STATUS(s, setSVCoefficients(nSV, model));
        DAAL_CHECK_STATUS(s, setSVIndices(nSV, model));
        if (xTable.getDataLayout() == NumericTableIface::csrArray)
        {
            DAAL_CHECK_STATUS(s, setSV_CSR(model, xTable, nSV));
        }
        else
        {
            DAAL_CHECK_STATUS(s, setSV_Dense(model, xTable, nSV));
        }
        /* Calculate bias and write it into model */
        model.setBias(double(calculateBias(cw)));
        return s;
    }

protected:
    /**
     * \brief Write classification coefficients into resulting model
     *template <typename algorithmFPType, typename ParameterType, CpuType cpu>
     * \param[in]  nSV          Number of support vectors
     * \param[out] model        Resulting model
     */
    services::Status setSVCoefficients(size_t nSV, Model & model) const
    {
        const algorithmFPType zero(0.0);
        NumericTablePtr svCoeffTable = model.getClassificationCoefficients();
        services::Status s;
        DAAL_CHECK_STATUS(s, svCoeffTable->resize(nSV));

        WriteOnlyRows<algorithmFPType, cpu> mtSvCoeff(*svCoeffTable, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSvCoeff);
        algorithmFPType * const svCoeff = mtSvCoeff.get();

        for (size_t i = 0, iSV = 0; i < _nVectors; ++i)
        {
            if (_alpha[i] != zero)
            {
                svCoeff[iSV] = _y[i] * _alpha[i];
                ++iSV;
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

        WriteOnlyRows<int, cpu> mtSvIndices(*svIndicesTable, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSvIndices);
        int * const svIndices = mtSvIndices.get();

        const algorithmFPType zero(0.0);
        for (size_t i = 0, iSV = 0; i < _nVectors; ++i)
        {
            if (_alpha[i] != zero)
            {
                DAAL_ASSERT(_cache->getDataRowIndex(i) <= services::internal::MaxVal<int>::get())
                svIndices[iSV] = (int)_cache->getDataRowIndex(i);
                ++iSV;
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
    services::Status setSV_Dense(Model & model, const NumericTable & xTable, size_t nSV) const
    {
        /* Allocate memory for support vectors and coefficients */
        NumericTablePtr svTable = model.getSupportVectors();
        services::Status s;
        DAAL_CHECK_STATUS(s, svTable->resize(nSV));
        if (nSV == 0) return s;

        WriteOnlyRows<algorithmFPType, cpu> mtSv(*svTable, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSv);
        algorithmFPType * const sv = mtSv.get();

        NumericTablePtr svIndicesTable = model.getSupportIndices();
        ReadRows<int, cpu> mtSvIndices(svIndicesTable.get(), 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSvIndices);
        const int * const svIndices = mtSvIndices.get();

        const size_t p = xTable.getNumberOfColumns();

        SafeStatus safeStat;
        daal::threader_for(nSV, nSV, [&](const size_t iBlock) {
            size_t iRows = svIndices[iBlock];
            ReadRows<algorithmFPType, cpu> mtX(const_cast<NumericTable *>(&xTable), iRows, 1);
            DAAL_CHECK_BLOCK_STATUS_THR(mtX);
            const algorithmFPType * const dataIn = mtX.get();
            algorithmFPType * dataOut            = sv + iBlock * p;
            DAAL_CHECK_THR(!services::internal::daal_memcpy_s(dataOut, p * sizeof(algorithmFPType), dataIn, p * sizeof(algorithmFPType)),
                           services::ErrorMemoryCopyFailedInternal);
        });
        return safeStat.detach();
    }

    /**
     * \brief Write support vectors in CSR format into resulting model
     *
     * \param[out] model        Resulting model
     * \param[in]  xTable       Input data set in CSR layout
     * \param[in]  nSV          Number of support vectors
     */
    services::Status setSV_CSR(Model & model, const NumericTable & xTable, size_t nSV) const
    {
        TArray<size_t, cpu> aSvRowOffsets(nSV + 1);
        DAAL_CHECK_MALLOC(aSvRowOffsets.get());
        size_t * const svRowOffsetsBuffer = aSvRowOffsets.get();

        CSRNumericTableIface * const csrIface = dynamic_cast<CSRNumericTableIface * const>(const_cast<NumericTable *>(&xTable));
        DAAL_CHECK(csrIface, services::ErrorEmptyCSRNumericTable);

        ReadRowsCSR<algorithmFPType, cpu> mtX;

        const algorithmFPType zero(0.0);
        /* Calculate row offsets for the table that stores support vectors */
        svRowOffsetsBuffer[0] = 1;
        for (size_t i = 0, iSV = 0; i < _nVectors; ++i)
        {
            if (_alpha[i] > zero)
            {
                const size_t rowIndex = _cache->getDataRowIndex(i);
                mtX.set(csrIface, rowIndex, 1);
                DAAL_CHECK_BLOCK_STATUS(mtX);
                svRowOffsetsBuffer[iSV + 1] = svRowOffsetsBuffer[iSV] + (mtX.rows()[1] - mtX.rows()[0]);
                iSV++;
            }
        }

        services::Status s;
        /* Allocate memory for storing support vectors and coefficients */
        CSRNumericTablePtr svTable = services::staticPointerCast<CSRNumericTable, NumericTable>(model.getSupportVectors());
        DAAL_CHECK_STATUS(s, svTable->resize(nSV));
        if (nSV == 0) return s;

        const size_t svDataSize = svRowOffsetsBuffer[nSV] - svRowOffsetsBuffer[0];
        DAAL_CHECK_STATUS(s, svTable->allocateDataMemory(svDataSize));

        /* Copy row offsets into the table */
        size_t * svRowOffsets = nullptr;
        svTable->getArrays<algorithmFPType>(NULL, NULL, &svRowOffsets);
        for (size_t i = 0; i < nSV + 1; i++)
        {
            svRowOffsets[i] = svRowOffsetsBuffer[i];
        }

        WriteOnlyRowsCSR<algorithmFPType, cpu> mtSv(*svTable, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSv);
        algorithmFPType * sv        = mtSv.values();
        size_t * const svColIndices = mtSv.cols();

        for (size_t i = 0, iSV = 0, svOffset = 0; i < _nVectors; ++i)
        {
            if (_alpha[i] == zero) continue;
            const size_t rowIndex = _cache->getDataRowIndex(i);
            mtX.set(csrIface, rowIndex, 1);
            DAAL_CHECK_BLOCK_STATUS(mtX);
            const algorithmFPType * const xi  = mtX.values();
            const size_t * const xiColIndices = mtX.cols();
            const size_t nNonZeroValuesInRow  = mtX.rows()[1] - mtX.rows()[0];
            for (size_t j = 0; j < nNonZeroValuesInRow; j++, svOffset++)
            {
                sv[svOffset]           = xi[j];
                svColIndices[svOffset] = xiColIndices[j];
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
    algorithmFPType calculateBias(const algorithmFPType * cw) const
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(saveResult.calculateBias);

        algorithmFPType bias    = algorithmFPType(0.0);
        size_t nGrad            = 0;
        algorithmFPType sumGrad = algorithmFPType(0.0);

        const algorithmFPType fpMax = MaxVal<algorithmFPType>::get();
        algorithmFPType ub          = fpMax;
        algorithmFPType lb          = -fpMax;

        for (size_t i = 0; i < _nVectors; ++i)
        {
            const algorithmFPType gradi = _grad[i];
            const algorithmFPType yi    = _y[i];
            const algorithmFPType cwi   = cw[i];
            const algorithmFPType ai    = _alpha[i];

            /* free SV: (0 < alpha < C)*/
            if (0 < ai && ai < cw[i])
            {
                sumGrad += gradi;
                ++nGrad;
            }
            if (HelperTrainSVM<algorithmFPType, cpu>::isUpper(yi, ai, cwi))
            {
                ub = services::internal::min<cpu, algorithmFPType>(ub, gradi);
            }
            if (HelperTrainSVM<algorithmFPType, cpu>::isLower(yi, ai, cwi))
            {
                lb = services::internal::max<cpu, algorithmFPType>(lb, gradi);
            }
        }
        if (nGrad == 0)
        {
            bias = -0.5 * (ub + lb);
        }
        else
        {
            bias = -sumGrad / algorithmFPType(nGrad);
        }

        return bias;
    }

private:
    const size_t _nVectors;                             //Number of observations in the input data set
    const algorithmFPType * _y;                         //Array of class labels
    const algorithmFPType * _alpha;                     //Array of classification coefficients
    const algorithmFPType * _grad;                      //Array of classification coefficients
    SVMCacheCommonIface<algorithmFPType, cpu> * _cache; //caches matrix Q (kernel(x[i], x[j])) values
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

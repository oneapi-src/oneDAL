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

    services::Status compute(const NumericTable & xTable, Model & model, algorithmFPType C) const
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(saveResult);

        const algorithmFPType zero(0.0);
        size_t nSV = 0;
        for (size_t i = 0; i < _nVectors; i++)
        {
            if (_alpha[i] > zero) nSV++;
        }
        printf(">> nSV %lu\n", nSV);

        model.setNFeatures(xTable.getNumberOfColumns());
        services::Status s;
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
        model.setBias(double(calculateBias(C)));
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
        DAAL_ITTNOTIFY_SCOPED_TASK(saveResult.setSVCoefficients);

        const algorithmFPType zero(0.0);
        NumericTablePtr svCoeffTable = model.getClassificationCoefficients();
        services::Status s;
        DAAL_CHECK_STATUS(s, svCoeffTable->resize(nSV));

        WriteOnlyRows<algorithmFPType, cpu> mtSvCoeff(*svCoeffTable, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSvCoeff);
        algorithmFPType * svCoeff = mtSvCoeff.get();

        for (size_t i = 0, iSV = 0; i < _nVectors; i++)
        {
            if (_alpha[i] != zero)
            {
                svCoeff[iSV] = _y[i] * _alpha[i];
                // svCoeff[iSV] = _alpha[i];
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
        DAAL_ITTNOTIFY_SCOPED_TASK(saveResult.setSVIndices);

        NumericTablePtr svIndicesTable = model.getSupportIndices();
        services::Status s;
        DAAL_CHECK_STATUS(s, svIndicesTable->resize(nSV));

        WriteOnlyRows<int, cpu> mtSvIndices(*svIndicesTable, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSvIndices);
        int * svIndices = mtSvIndices.get();

        const algorithmFPType zero(0.0);
        for (size_t i = 0, iSV = 0; i < _nVectors; i++)
        {
            if (_alpha[i] != zero)
            {
                DAAL_ASSERT(_cache->getDataRowIndex(i) <= services::internal::MaxVal<int>::get())
                svIndices[iSV] = (int)_cache->getDataRowIndex(i);
                iSV++;
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
        DAAL_ITTNOTIFY_SCOPED_TASK(saveResult.setSV_Dense);

        const size_t nFeatures = xTable.getNumberOfColumns();
        /* Allocate memory for support vectors and coefficients */
        NumericTablePtr svTable = model.getSupportVectors();
        services::Status s;
        DAAL_CHECK_STATUS(s, svTable->resize(nSV));
        if (nSV == 0) return s;

        WriteOnlyRows<algorithmFPType, cpu> mtSv(*svTable, 0, nSV);
        DAAL_CHECK_BLOCK_STATUS(mtSv);
        algorithmFPType * sv = mtSv.get();

        const algorithmFPType zero(0.0);
        ReadRows<algorithmFPType, cpu> mtX;
        for (size_t i = 0, iSV = 0; i < _nVectors; i++)
        {
            if (_alpha[i] == zero) continue;
            const size_t rowIndex = _cache->getDataRowIndex(i);
            mtX.set(const_cast<NumericTable *>(&xTable), rowIndex, 1);
            DAAL_CHECK_BLOCK_STATUS(mtX);
            const algorithmFPType * xi = mtX.get();
            for (size_t j = 0; j < nFeatures; j++)
            {
                sv[iSV * nFeatures + j] = xi[j];
            }
            iSV++;
        }
        return s;
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
        DAAL_ITTNOTIFY_SCOPED_TASK(saveResult.setSV_CSR);

        TArray<size_t, cpu> aSvRowOffsets(nSV + 1);
        DAAL_CHECK_MALLOC(aSvRowOffsets.get());
        size_t * svRowOffsetsBuffer = aSvRowOffsets.get();

        CSRNumericTableIface * csrIface = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(&xTable));
        ReadRowsCSR<algorithmFPType, cpu> mtX;

        const algorithmFPType zero(0.0);
        /* Calculate row offsets for the table that stores support vectors */
        svRowOffsetsBuffer[0] = 1;
        for (size_t i = 0, iSV = 0; i < _nVectors; i++)
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
        algorithmFPType * sv  = mtSv.values();
        size_t * svColIndices = mtSv.cols();

        for (size_t i = 0, iSV = 0, svOffset = 0; i < _nVectors; i++)
        {
            if (_alpha[i] == zero) continue;
            const size_t rowIndex = _cache->getDataRowIndex(i);
            mtX.set(csrIface, rowIndex, 1);
            DAAL_CHECK_BLOCK_STATUS(mtX);
            const algorithmFPType * xi       = mtX.values();
            const size_t * xiColIndices      = mtX.cols();
            const size_t nNonZeroValuesInRow = mtX.rows()[1] - mtX.rows()[0];
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
    algorithmFPType calculateBias(algorithmFPType C) const
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(saveResult.calculateBias);

        algorithmFPType bias;
        const algorithmFPType zero(0.0);
        const algorithmFPType one(1.0);
        size_t num_yg          = 0;
        algorithmFPType sum_yg = 0.0;

        const algorithmFPType fpMax = MaxVal<algorithmFPType>::get();
        // algorithmFPType ub          = -(fpMax);
        // algorithmFPType lb          = fpMax;

        algorithmFPType ub = fpMax;
        algorithmFPType lb = -fpMax;

        for (size_t i = 0; i < _nVectors; i++)
        {
            const algorithmFPType gradi      = _grad[i];
            const algorithmFPType dualCoeffi = _y[i] * _alpha[i];

            /* free SV: (0 < alpha < C)*/
            if (0 < dualCoeffi && dualCoeffi < C)
            {
                sum_yg += gradi;
                num_yg++;
            }
            else
            {
                if (isUpper(_y[i], dualCoeffi, C))
                {
                    ub = services::internal::min<cpu, algorithmFPType>(ub, gradi);
                }
                if (isLower(_y[i], dualCoeffi, C))
                {
                    lb = services::internal::max<cpu, algorithmFPType>(lb, gradi);
                }
            }
        }

        if (num_yg == 0)
        {
            bias = -0.5 * (ub + lb);
        }
        else
        {
            bias = -sum_yg / (algorithmFPType)num_yg;
        }

        return bias;
    }

private:
    static bool isUpper(const algorithmFPType y, const algorithmFPType alpha, const algorithmFPType C)
    {
        return (y > 0 && alpha < C) || (y < 0 && alpha > 0);
    }
    static bool isLower(const algorithmFPType y, const algorithmFPType alpha, const algorithmFPType C)
    {
        return (y > 0 && alpha > 0) || (y < 0 && alpha < C);
    }

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

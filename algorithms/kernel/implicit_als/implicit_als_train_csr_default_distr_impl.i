/* file: implicit_als_train_csr_default_distr_impl.i */
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

/*
//++
//  Implementation of impicit ALS training algorithm for distributed processing mode
//--
*/

#ifndef __IMPLICIT_ALS_TRAIN_CSR_DEFAULT_DISTR_IMPL_I__
#define __IMPLICIT_ALS_TRAIN_CSR_DEFAULT_DISTR_IMPL_I__

#include "service_micro_table.h"
#include "service_memory.h"
#include "service_blas.h"

using namespace daal::services::internal;
using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace internal
{
template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainDistrStep1Kernel<algorithmFPType, cpu>::compute(
            implicit_als::PartialModel *partialModel, data_management::NumericTable *crossProduct,
            const Parameter *parameter)
{
    const size_t maxBlockSize = 100 * 1024 * 1024;

    size_t nFactors = parameter->nFactors;
    size_t nRowsInBlock = maxBlockSize / nFactors;

    daal::internal::BlockMicroTable<algorithmFPType, readOnly,  cpu> mtFactors(partialModel->getFactors().get());
    size_t nRows = mtFactors.getFullNumberOfRows();
    size_t nBlocks = nRows / nRowsInBlock;
    if (nBlocks * nRowsInBlock < nRows) { nBlocks++; }

    if (nBlocks == 1) { nRowsInBlock = nRows; }

    size_t nRowsRead = 0;

    daal::internal::BlockMicroTable<algorithmFPType, writeOnly,  cpu> mtCrossProduct(crossProduct);
    algorithmFPType *cp;
    nRowsRead = mtCrossProduct.getBlockOfRows(0, nFactors, &cp);
    if (nRowsRead < nFactors)
    { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

    algorithmFPType zero = (algorithmFPType)0.0;
    for (size_t i = 0; i < nFactors * nFactors; i++)
    {
        cp[i] = zero;
    }

    /* SYRK parameters */
    char uplo = 'U';
    char trans = 'N';
    algorithmFPType alpha = 1.0;
    algorithmFPType beta  = 1.0;

    algorithmFPType *srcFactorsBlock;
    for (size_t block = 0; block < nBlocks; block++)
    {
        size_t iStart = block * nRowsInBlock;
        size_t iEnd   = iStart + nRowsInBlock;
        if (iEnd > nRows) iEnd = nRows;
        size_t nRowsToCP = iEnd - iStart;
        nRowsRead = mtFactors.getBlockOfRows(iStart, nRowsToCP, &srcFactorsBlock);
        if (nRowsRead < nRowsToCP)
        { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

        Blas<algorithmFPType, cpu>::xsyrk(&uplo, &trans, (DAAL_INT *)&nFactors, (DAAL_INT *)&nRowsToCP, &alpha, srcFactorsBlock,
                (DAAL_INT *)&nFactors, &beta, cp, (DAAL_INT *)&nFactors);
        mtFactors.release();
    }

    mtCrossProduct.release();
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainDistrStep2Kernel<algorithmFPType, cpu>::compute(
            size_t nParts, data_management::NumericTable **partialCrossProducts,
            data_management::NumericTable *crossProduct, const Parameter *parameter)
{
    size_t nFactors = parameter->nFactors;

    daal::internal::BlockMicroTable<algorithmFPType, writeOnly,  cpu> mtCrossProduct(crossProduct);
    algorithmFPType *cp;
    size_t nRowsRead = mtCrossProduct.getBlockOfRows(0, nFactors, &cp);
    if (nRowsRead < nFactors)
    { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

    algorithmFPType zero = (algorithmFPType)0.0;
    for (size_t i = 0; i < nFactors * nFactors; i++)
    {
        cp[i] = zero;
    }

    for (size_t i = 0; i < nParts; i++)
    {
         daal::internal::BlockMicroTable<algorithmFPType, readOnly,  cpu> mtPartialCrossProduct(
                partialCrossProducts[i]);
        algorithmFPType *partialCP;
        size_t nRowsRead = mtPartialCrossProduct.getBlockOfRows(0, nFactors, &partialCP);
        if (nRowsRead < nFactors)
        { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
        for (size_t j = 0; j < nFactors * nFactors; j++)
        {
            cp[j] += partialCP[j];
        }
    }
    mtCrossProduct.release();
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainDistrStep3Kernel<algorithmFPType, cpu>::compute(
            implicit_als::PartialModel *srcPartialModel, data_management::NumericTable *offsetTable,
            data_management::KeyValueDataCollection *dstPartialModels, const Parameter *parameter)
{
    int offset = 0;
    daal::internal::BlockMicroTable<int, readOnly, cpu> mtOffset(offsetTable);
    int *offsetData;
    size_t nRowsRead = mtOffset.getBlockOfRows(0, 1, &offsetData);
    if (nRowsRead < 1)
    { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

    offset = offsetData[0];
    mtOffset.release();

    size_t nBlocks = dstPartialModels->size();
    size_t nFactors = parameter->nFactors;

    daal::threader_for(nBlocks, nBlocks, [ & ](size_t i)
    {
        PartialModel *dstPartialModel = static_cast<PartialModel *>((*dstPartialModels)[i].get());
        daal::internal::BlockMicroTable<algorithmFPType, readOnly,  cpu> mtSrcFactors(srcPartialModel->getFactors().get());
        daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu> mtDstFactors(dstPartialModel->getFactors().get());
        daal::internal::BlockMicroTable<int, readOnly, cpu> dstIndices(dstPartialModel->getIndices().get());

        size_t nRows = mtDstFactors.getFullNumberOfRows();
        int *indices;
        size_t nRowsReadLocal = dstIndices.getBlockOfRows(0, nRows, &indices);
        if (nRowsReadLocal < nRows)
        { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
        for (size_t j = 0; j < nRows; j++)
        {
            algorithmFPType *srcFactor;
            nRowsReadLocal = mtSrcFactors.getBlockOfRows(indices[j] - offset, 1, &srcFactor);
            if (nRowsReadLocal < 1)
            { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

            algorithmFPType *dstFactor;
            nRowsReadLocal = mtDstFactors.getBlockOfRows(j, 1, &dstFactor);
            if (nRowsReadLocal < 1)
            { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

            daal::services::daal_memcpy_s(dstFactor, nFactors * sizeof(algorithmFPType),
                                          srcFactor, nFactors * sizeof(algorithmFPType));
            mtSrcFactors.release();
            mtDstFactors.release();
        }
        dstIndices.release();
    } );
}

template <typename algorithmFPType, CpuType cpu>
struct AlsTls
{
    daal::internal::BlockMicroTable<algorithmFPType, readOnly,  cpu> **mtSrcFactors;
    daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu>  *mtDstFactors;
    algorithmFPType *lhs;
    size_t nBlocks;
    services::SharedPtr<services::Error> localError;

    AlsTls(size_t nBlocks, size_t nFactors, data_management::KeyValueDataCollection *srcPartialModels,
           implicit_als::PartialModel *dstPartialModel) : nBlocks(nBlocks), localError(new services::Error())
    {
        lhs = (algorithmFPType *)daal::services::daal_malloc(nFactors * nFactors * sizeof(algorithmFPType));

        mtDstFactors = new daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu>(dstPartialModel->getFactors().get());

        mtSrcFactors = (daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> **)daal::services::daal_malloc(
                nBlocks * sizeof(daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> *));

        for (size_t i = 0; i < nBlocks; i++)
        {
            PartialModel *srcPartialModel = static_cast<PartialModel *>((*srcPartialModels)[i].get());
            mtSrcFactors[i] = new daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu>(
                    srcPartialModel->getFactors().get());
        }
    }

    ~AlsTls()
    {
        daal::services::daal_free(lhs);
        delete mtDstFactors;
        for (size_t i = 0; i < nBlocks; i++)
        {
            delete mtSrcFactors[i];
        }
        daal::services::daal_free(mtSrcFactors);
    }
};

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainDistrStep4Kernel<algorithmFPType, fastCSR, cpu>::compute(
            data_management::KeyValueDataCollection *srcPartialModels, data_management::NumericTable *dataTable,
            data_management::NumericTable *cpTable, implicit_als::PartialModel *dstPartialModel,
            const Parameter *parameter)
{
    size_t nBlocks = srcPartialModels->size();
    size_t *nFactorsRows = (size_t *)daal::services::daal_malloc(nBlocks * sizeof(size_t));
    int **indices = (int **)daal::services::daal_malloc(nBlocks * sizeof(int *));
    daal::internal::BlockMicroTable<int, readOnly, cpu> **mtIndices =
            (daal::internal::BlockMicroTable<int, readOnly, cpu> **)daal::services::daal_malloc(
                    nBlocks * sizeof(daal::internal::BlockMicroTable<int, readOnly, cpu> *));

    if (!nFactorsRows || !indices || !mtIndices)
    { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }

    for (size_t i = 0; i < nBlocks; i++)
    {
        PartialModel *srcPartialModel = static_cast<PartialModel *>((*srcPartialModels)[i].get());
        mtIndices[i] = new daal::internal::BlockMicroTable<int, readOnly, cpu>(srcPartialModel->getIndices().get());

        size_t nRows = mtIndices[i]->getFullNumberOfRows();
        nFactorsRows[i] = nRows;

        size_t nRowsRead = mtIndices[i]->getBlockOfRows(0, nRows, &(indices[i]));
        if (nRowsRead < nRows)
        { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }
    }

    /* Compute resulting partial factors */
    size_t nFactors = parameter->nFactors;
    algorithmFPType alpha  = (algorithmFPType)(parameter->alpha);
    algorithmFPType lambda = (algorithmFPType)(parameter->lambda);

    daal::tls<AlsTls<algorithmFPType, cpu> *> alsTls([=]()
    {
        return new AlsTls<algorithmFPType, cpu>(nBlocks, nFactors, srcPartialModels, dstPartialModel);
    });

    daal::internal::BlockMicroTable<algorithmFPType, readOnly,  cpu> mtXTX(cpTable);
    algorithmFPType *xtx;
    size_t nRowsRead = mtXTX.getBlockOfRows(0, nFactors, &xtx);
    if (nRowsRead < nFactors)
    { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

    daal::internal::CSRBlockMicroTable<algorithmFPType, readOnly, cpu> mtData(dataTable);
    size_t nRows = mtData.getFullNumberOfRows();
    algorithmFPType *data;
    size_t *colIndices, *rowOffsets;

    nRowsRead = mtData.getSparseBlock(0, nRows, &data, &colIndices, &rowOffsets);
    if (nRowsRead < nRows)
    { this->_errors->add(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return; }

    daal::threader_for(nRows, nRows, [ & ](size_t i)
    {
        AlsTls<algorithmFPType, cpu> *alsTlsLocal = alsTls.local();
        daal::internal::BlockMicroTable<algorithmFPType, writeOnly, cpu>  *mtDstFactors = alsTlsLocal->mtDstFactors;
        daal::internal::BlockMicroTable<algorithmFPType, readOnly,  cpu> **mtSrcFactors = alsTlsLocal->mtSrcFactors;
        services::SharedPtr<services::Error> &localError = alsTlsLocal->localError;
        algorithmFPType *lhs = alsTlsLocal->lhs;
        algorithmFPType *rhs;
        size_t nRowsReadLocal = mtDstFactors->getBlockOfRows(i, 1, &rhs);
        if (nRowsReadLocal < 1)
        {
            localError->setId(services::ErrorIncorrectNumberOfRowsInInputNumericTable); return;
        }

        service_memset<algorithmFPType, cpu>(rhs, 0.0, nFactors);
        daal::services::daal_memcpy_s(lhs, nFactors * nFactors * sizeof(algorithmFPType),
                                      xtx, nFactors * nFactors * sizeof(algorithmFPType));

        formSystem(i, data, colIndices, rowOffsets, nFactors, nBlocks, nFactorsRows, indices, mtSrcFactors,
                alpha, lhs, rhs, lambda, localError);
        if(localError->id() != services::NoErrorMessageFound) return;

        /* Solve system of normal equations */
        this->solve(&nFactors, lhs, &nFactors, rhs, &nFactors);
        if(localError->id() != services::NoErrorMessageFound) return;
        mtDstFactors->release();
    } );

    alsTls.reduce([ = ](AlsTls<algorithmFPType, cpu> *alsTlsLocal)
    {
        if(alsTlsLocal->localError->id() != services::NoErrorMessageFound)
        {
            this->_errors->add(alsTlsLocal->localError);
        }
        delete alsTlsLocal;
    } );

    for (size_t i = 0; i < nBlocks;  i++)
    {
        mtIndices[i]->release();
        delete mtIndices[i];
    }

    mtData.release();
    mtXTX.release();
    daal::services::daal_free(nFactorsRows);
    daal::services::daal_free(indices);
    daal::services::daal_free(mtIndices);
}

template <typename algorithmFPType, CpuType cpu>
void ImplicitALSTrainDistrStep4Kernel<algorithmFPType, fastCSR, cpu>::formSystem(
            size_t i, algorithmFPType *data, size_t *colIndices, size_t *rowOffsets,
            size_t nFactors, size_t nBlocks, size_t *nColFactorsRows,
            int **indices,
            daal::internal::BlockMicroTable<algorithmFPType, readOnly, cpu> **mtSrcFactors,
            algorithmFPType alpha, algorithmFPType *lhs, algorithmFPType *rhs, algorithmFPType lambda,
            services::SharedPtr<services::Error> &error)
{
    size_t startIdx = rowOffsets[i]   - 1;
    size_t endIdx   = rowOffsets[i + 1] - 1;

    /* Update the linear system of normal equations */
    for (size_t j = startIdx; j < endIdx; j++)
    {
        algorithmFPType c1 = alpha * data[j];
        algorithmFPType c = c1 + 1.0;
        int colIndex = (int)colIndices[j] - 1;

        int blockIndex = -1;
        /* find block that contains needed index */
        for (size_t block = 0; block < nBlocks; block++)
        {
            if (indices[block] && indices[block][0] <= colIndex && colIndex <= indices[block][nColFactorsRows[block] - 1])
            {
                blockIndex = block;
                break;
            }
        }
        if (blockIndex == -1)
        {
            error->setId(services::ErrorALSInconsistentSparseDataBlocks);
            return;
        }

        int *blockIndices = indices[blockIndex];
        /* find index in the block using binary search */
        size_t hiIndex = nColFactorsRows[blockIndex] - 1;
        size_t loIndex = 0;
        size_t meIndex = ((loIndex + hiIndex) >> 1);
        while (colIndex != blockIndices[meIndex])
        {
            if (colIndex < blockIndices[meIndex]) { hiIndex = meIndex - 1; }
            if (colIndex > blockIndices[meIndex]) { loIndex = meIndex + 1; }
            meIndex = ((loIndex + hiIndex) >> 1);
            if (loIndex >= hiIndex) { break; }
        }
        if (colIndex != blockIndices[meIndex])
        {
            error->setId(services::ErrorALSInconsistentSparseDataBlocks);
            return;
        }

        algorithmFPType *colFactorsRow;
        size_t nRowsRead = mtSrcFactors[blockIndex]->getBlockOfRows(meIndex, 1, &colFactorsRow);
        if (nRowsRead < 1)
        {
            error->setId(services::ErrorIncorrectNumberOfRowsInInputNumericTable);
            return;
        }

        this->updateSystem(&nFactors, colFactorsRow, &c1, &c, lhs, &nFactors, rhs);
        mtSrcFactors[blockIndex]->release();
    }

    /* Add regularization term */
    algorithmFPType gamma = lambda * (endIdx - startIdx);
    for (size_t k = 0; k < nFactors; k++)
    {
        lhs[k * nFactors + k] += gamma;
    }
}

}
}
}
}
}

#endif

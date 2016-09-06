/* file: kernel_function_csr_base.h */
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
//  Declaration of template structs that calculate SVM Kernel functions.
//--
*/

#ifndef __KERNEL_FUNCTION_CSR_BASE_H__
#define __KERNEL_FUNCTION_CSR_BASE_H__

#include "numeric_table.h"
#include "kernel_function_types_linear.h"
#include "kernel_function_types_rbf.h"
#include "kernel_function_linear.h"
#include "kernel_function_rbf.h"
#include "service_micro_table.h"
#include "kernel.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace internal
{

template <typename AlgorithmFPType, CpuType cpu>
struct KernelCSRImplBase : public Kernel
{
    void prepareDataVectorVector(
        CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA1,
        CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA2,
        BlockMicroTable<AlgorithmFPType, writeOnly, cpu> &mtR,
        const ParameterBase *svmPar,
        size_t *nVectors1, AlgorithmFPType **dataA1, size_t **colIndicesA1, size_t **rowOffsetsA1,
        size_t *nVectors2, AlgorithmFPType **dataA2, size_t **colIndicesA2, size_t **rowOffsetsA2,
        AlgorithmFPType **dataR, bool inputTablesSame)
    {
        mtA1.getSparseBlock(svmPar->rowIndexX, 1, dataA1, colIndicesA1, rowOffsetsA1);
        mtA2.getSparseBlock(svmPar->rowIndexY, 1, dataA2, colIndicesA2, rowOffsetsA2);
        mtR. getBlockOfRows(svmPar->rowIndexResult, 1, dataR);
    }

    void prepareDataMatrixVector(
        CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA1,
        CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA2,
        BlockMicroTable<AlgorithmFPType, writeOnly, cpu> &mtR,
        const ParameterBase *svmPar,
        size_t *nVectors1, AlgorithmFPType **dataA1, size_t **colIndicesA1, size_t **rowOffsetsA1,
        size_t *nVectors2, AlgorithmFPType **dataA2, size_t **colIndicesA2, size_t **rowOffsetsA2,
        AlgorithmFPType **dataR, bool inputTablesSame)
    {
        size_t n = mtA1.getFullNumberOfRows();
        mtA1.getSparseBlock(0, n, dataA1, colIndicesA1, rowOffsetsA1);
        mtA2.getSparseBlock(svmPar->rowIndexY, 1, dataA2, colIndicesA2, rowOffsetsA2);
        mtR. getBlockOfRows(svmPar->rowIndexResult, 1, dataR);
        *nVectors1 = n;
    }

    void prepareDataMatrixMatrix(
        CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA1,
        CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA2,
        BlockMicroTable<AlgorithmFPType, writeOnly, cpu> &mtR,
        const ParameterBase *svmPar,
        size_t *nVectors1, AlgorithmFPType **dataA1, size_t **colIndicesA1, size_t **rowOffsetsA1,
        size_t *nVectors2, AlgorithmFPType **dataA2, size_t **colIndicesA2, size_t **rowOffsetsA2,
        AlgorithmFPType **dataR, bool inputTablesSame)
    {
        size_t n1 = mtA1.getFullNumberOfRows();
        size_t n2 = mtA2.getFullNumberOfRows();
        mtA1.getSparseBlock(0, n1, dataA1, colIndicesA1, rowOffsetsA1);
        if (!inputTablesSame) mtA2.getSparseBlock(0, n2, dataA2, colIndicesA2, rowOffsetsA2);
        mtR. getBlockOfRows(0, n1, dataR);
        *nVectors1 = n1;
        *nVectors2 = n2;
    }

    virtual void computeInternal(size_t nFeatures,
                                 size_t nVectors1, const AlgorithmFPType *dataA1, const size_t *colIndicesA1, const size_t *rowOffsetsA1,
                                 size_t nVectors2, const AlgorithmFPType *dataA2, const size_t *colIndicesA2, const size_t *rowOffsetsA2,
                                 AlgorithmFPType *dataR, const ParameterBase *par, bool inputTablesSame) = 0;

    virtual void prepareData(CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA1,
                             CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> &mtA2,
                             BlockMicroTable<AlgorithmFPType, writeOnly, cpu> &mtR,
                             const ParameterBase *svmPar,
                             size_t *nVectors1, AlgorithmFPType **dataA1, size_t **colIndicesA1, size_t **rowOffsetsA1,
                             size_t *nVectors2, AlgorithmFPType **dataA2, size_t **colIndicesA2, size_t **rowOffsetsA2,
                             AlgorithmFPType **dataR, bool inputTablesSame) = 0;

    void compute(ComputationMode computationMode, const NumericTable *a1, const NumericTable *a2, NumericTable *r,
                 const daal::algorithms::Parameter *par)
    {

        _computationMode = computationMode;

        CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> mtA1(a1);
        CSRBlockMicroTable<AlgorithmFPType, readOnly,  cpu> mtA2(a2);
        BlockMicroTable<AlgorithmFPType, writeOnly, cpu> mtR(r);
        size_t nFeatures = mtA1.getFullNumberOfColumns();
        size_t nVectors1, nVectors2;
        AlgorithmFPType *dataA1, *dataA2, *dataR;
        size_t *colIndicesA1, *rowOffsetsA1, *colIndicesA2, *rowOffsetsA2;
        const ParameterBase *svmPar = static_cast<const ParameterBase *>(par);

        bool inputTablesSame = ((a1 == a2) ? true : false);

        prepareData(mtA1, mtA2, mtR, svmPar, &nVectors1, &dataA1, &colIndicesA1, &rowOffsetsA1,
                    &nVectors2, &dataA2, &colIndicesA2, &rowOffsetsA2, &dataR, inputTablesSame);
        computeInternal(nFeatures, nVectors1, dataA1, colIndicesA1, rowOffsetsA1,
                        nVectors2, dataA2, colIndicesA2, rowOffsetsA2, dataR, svmPar, inputTablesSame);

        mtA1.release();
        mtA2.release();
        mtR. release();
    }

protected:
    inline AlgorithmFPType computeDotProduct(size_t startIndex1, size_t endIndex1, const AlgorithmFPType *dataA1, const size_t *colIndicesA1,
                                             size_t startIndex2, size_t endIndex2, const AlgorithmFPType *dataA2, const size_t *colIndicesA2);

    ComputationMode _computationMode;
};

} // namespace internal

} // namespace kernel_function

} // namespace algorithms

} // namespace daal

#endif

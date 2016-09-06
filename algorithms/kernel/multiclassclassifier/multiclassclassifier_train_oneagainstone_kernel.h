/* file: multiclassclassifier_train_oneagainstone_kernel.h */
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
//  Declaration of template structs for One-Against-One method for Multi-class classifier
//  training algorithm for CSR input data.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_TRAIN_ONEAGAINSTONE_KERNEL_H__
#define __MULTICLASSCLASSIFIER_TRAIN_ONEAGAINSTONE_KERNEL_H__

#include "multi_class_classifier_model.h"

#include "threading.h"
#include "service_sort.h"
#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::data_management;

namespace daal
{
namespace algorithms
{
namespace multi_class_classifier
{
namespace training
{
namespace internal
{

template<typename algorithmFPType, CpuType cpu>
struct MultiClassClassifierTls
{
    MultiClassClassifierTls(size_t nFeatures, size_t nSubsetVectors, size_t dataSize, const NumericTable *xTable,
                            services::SharedPtr<classifier::training::Batch> simpleTraining)
    {
        subsetX = (algorithmFPType *)daal::services::daal_malloc((dataSize + nSubsetVectors) * sizeof(algorithmFPType));
        if (!subsetX) { error.setId(services::ErrorMemoryAllocationFailed); return; }
        subsetY = subsetX + dataSize;
        subsetYTable = NumericTablePtr(
                new HomogenNumericTableCPU<algorithmFPType, cpu> (subsetY, 1, nSubsetVectors));
        if (!subsetYTable) { error.setId(services::ErrorMemoryAllocationFailed); return; }
        this->simpleTraining = simpleTraining->clone();
        if (xTable->getDataLayout() == NumericTableIface::csrArray)
        {
            colIndicesX = (size_t *)daal::services::daal_malloc((dataSize + nSubsetVectors + 1) * sizeof(algorithmFPType));
            if (!colIndicesX) { error.setId(services::ErrorMemoryAllocationFailed); return; }
            rowOffsetsX = colIndicesX + dataSize;
            subsetXTable = NumericTablePtr(
                    new CSRNumericTable(subsetX, colIndicesX, rowOffsetsX, nFeatures));
            mtX = new CSRBlockMicroTable<algorithmFPType, readOnly, cpu>(xTable);
        }
        else
        {
            colIndicesX = NULL;
            subsetXTable = NumericTablePtr(
                new HomogenNumericTableCPU<algorithmFPType, cpu> (subsetX, nFeatures, nSubsetVectors));
            mtX = new BlockMicroTable<algorithmFPType, readOnly, cpu>(xTable);
        }
        if (!subsetXTable || !mtX) { error.setId(services::ErrorMemoryAllocationFailed); return; }
    }

    virtual ~MultiClassClassifierTls()
    {
        daal::services::daal_free(subsetX);
        if (colIndicesX) daal::services::daal_free(colIndicesX);
        delete mtX;
    }

    algorithmFPType *subsetX;
    algorithmFPType *subsetY;
    size_t *colIndicesX;
    size_t *rowOffsetsX;
    NumericTablePtr subsetXTable;
    NumericTablePtr subsetYTable;
    MicroTable *mtX;
    services::SharedPtr<classifier::training::Batch> simpleTraining;
    services::Error error;
};

template<typename algorithmFPType, CpuType cpu>
struct MultiClassClassifierTrainKernel<oneAgainstOne, algorithmFPType, cpu> : public Kernel
{
    void compute(const NumericTable *xTable, const NumericTable *yTable, daal::algorithms::Model *r,
                 const daal::algorithms::Parameter *par);

protected:
    void computeDataSize(size_t nVectors, size_t nFeatures, size_t nClasses,
                const NumericTable *xTable, int *y, size_t *nSubsetVectorsPtr, size_t *dataSizePtr);

    void copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label,
                              BlockMicroTable  <algorithmFPType, readOnly, cpu> &mtX,
                              const int *y, algorithmFPType *subsetX, algorithmFPType *subsetY,
                              size_t *nRowsPtr, size_t *dataSize);

    void copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label,
                              CSRBlockMicroTable<algorithmFPType, readOnly, cpu> &mtX,
                              const int *y, algorithmFPType *subsetX, size_t *colIndicesX, size_t *rowOffsetsX,
                              algorithmFPType *subsetY, size_t *nRowsPtr, size_t *dataSize);
};

} // namespace internal
} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif

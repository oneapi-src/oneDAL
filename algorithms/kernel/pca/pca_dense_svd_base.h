/* file: pca_dense_svd_base.h */
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
//  Declaration of template structs that calculate PCA SVD.
//--
*/

#ifndef __PCA_DENSE_SVD_BASE_H__
#define __PCA_DENSE_SVD_BASE_H__

namespace daal
{
namespace algorithms
{
namespace pca
{
namespace internal
{
enum InputDataType
{
    nonNormalizedDataset = 0,   /*!< Original, non-normalized data set */
    normalizedDataset    = 1,   /*!< Normalized data set whose feature vectors have zero average and unit variance */
    correlation          = 2    /*!< Correlation matrix */
};


template <typename interm, CpuType cpu>
class PCASVDKernelBase : public Kernel
{
public:
    PCASVDKernelBase() {};

    virtual ~PCASVDKernelBase() {}

    void setType(InputDataType type)
    {
        _type = type;
    }

protected:
    void scaleSingularValues(data_management::NumericTable *eigenvaluesTable, size_t nVectors);

    InputDataType _type;
};

template <typename interm, CpuType cpu>
void PCASVDKernelBase<interm, cpu>::scaleSingularValues(NumericTable *eigenvaluesTable, size_t nVectors)
{
    size_t nFeatures = eigenvaluesTable->getNumberOfColumns();

    interm *eigenvalues;
    BlockDescriptor<interm> block;
    eigenvaluesTable->getBlockOfRows(0, 1, data_management::readWrite, block);
    eigenvalues = block.getBlockPtr();

    for (size_t i = 0; i < nFeatures; i++)
    {
        eigenvalues[i] = eigenvalues[i] * eigenvalues[i] / (nVectors - 1);
    }

    eigenvaluesTable->releaseBlockOfRows(block);
}

} // namespace internal
} // namespace pca
} // namespace algorithms
} // namespace daal
#endif

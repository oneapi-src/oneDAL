/* file: multiclassclassifier_train_oneagainstone_impl.i */
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
//  Implementation of One-Against-One method for Multi-class classifier
//  training algorithm for CSR input data.
//--
*/

#ifndef __MULTICLASSCLASSIFIER_TRAIN_ONEAGAINSTONE_IMPL_I__
#define __MULTICLASSCLASSIFIER_TRAIN_ONEAGAINSTONE_IMPL_I__

#include "multi_class_classifier_model.h"

#include "threading.h"
#include "service_sort.h"
#include "service_memory.h"
#include "service_micro_table.h"
#include "service_numeric_table.h"

#include "service_blas.h"

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
void MultiClassClassifierTrainKernel<oneAgainstOne, algorithmFPType, cpu>::
    compute(const NumericTable *xTable, const NumericTable *yTable, daal::algorithms::Model *r,
            const daal::algorithms::Parameter *par)
{
    Model *model = static_cast<Model *>(r);
    const Parameter *mccPar = static_cast<const Parameter *>(par);

    services::SharedPtr<classifier::training::Batch> simpleTrainingInit = mccPar->training->clone();

    size_t nClasses = mccPar->nClasses;

    FeatureMicroTable<int, readOnly, cpu> mtY(yTable);
    size_t nFeatures = xTable->getNumberOfColumns();
    size_t nVectors  = xTable->getNumberOfRows();
    model->setNFeatures(nFeatures);

    int *y;
    mtY.getBlockOfColumnValues(0, 0, nVectors, &y);

    /* Compute data size needed to store the largest subset of input tables */
    size_t nSubsetVectors, dataSize;
    computeDataSize(nVectors, nFeatures, nClasses, xTable, y, &nSubsetVectors, &dataSize);

    /* Allocate memory for storing subsets of input data */
    daal::tls<MultiClassClassifierTls<algorithmFPType, cpu> *> subset([=]()
    {
        return new MultiClassClassifierTls<algorithmFPType, cpu>(
                nFeatures, nSubsetVectors, dataSize, xTable, simpleTrainingInit);
    } );

    size_t nModels = (nClasses * (nClasses - 1)) >> 1;
    daal::threader_for(nModels, nModels, [&](size_t imodel)
    {
        /* Find indices of positive and negative classes for current model */
        size_t i = 1;       /* index of the positive class */
        size_t j = 0;       /* index of the negative class */
        size_t isum = 0;    /* isum = (i*i - i) / 2; */
        while (i <= j || isum + j != imodel)
        {
            isum += i;
            i++;
            j = imodel - isum;
        }

        MultiClassClassifierTls<algorithmFPType, cpu> *subsetLocal = subset.local();
        if (subsetLocal->error.id() != services::NoErrorMessageFound) { return; }
        algorithmFPType *subsetX = subsetLocal->subsetX;
        size_t *colIndicesX = subsetLocal->colIndicesX;
        size_t *rowOffsetsX = subsetLocal->rowOffsetsX;
        algorithmFPType *subsetY = subsetLocal->subsetY;
        NumericTablePtr subsetXTable = subsetLocal->subsetXTable;
        NumericTablePtr subsetYTable = subsetLocal->subsetYTable;

        size_t nTotal, nPositive, nNegative;
        if (xTable->getDataLayout() == NumericTableIface::csrArray)
        {
            CSRBlockMicroTable<algorithmFPType, readOnly, cpu> *mtX =
                static_cast<CSRBlockMicroTable<algorithmFPType, readOnly, cpu> *>(subsetLocal->mtX);
            /* Prepare "positive" observations of the training subset */
            size_t positiveDataSize;
            rowOffsetsX[0] = 1;
            copyDataIntoSubtable(nFeatures, nVectors, i, 1, *mtX, y, subsetX, colIndicesX, rowOffsetsX,
                                 subsetY, &nPositive, &positiveDataSize);

            /* Prepare "negative" observations of the training subset */
            size_t negativeDataSize;
            copyDataIntoSubtable(nFeatures, nVectors, j, -1, *mtX, y,
                                 subsetX + positiveDataSize, colIndicesX + positiveDataSize, rowOffsetsX + nPositive,
                                 subsetY + nPositive, &nNegative, &negativeDataSize);
        }
        else
        {
            BlockMicroTable<algorithmFPType, readOnly, cpu> *mtX =
                static_cast<BlockMicroTable<algorithmFPType, readOnly, cpu> *>(subsetLocal->mtX);
            /* Prepare "positive" observations of the training subset */
            size_t positiveDataSize;
            copyDataIntoSubtable(nFeatures, nVectors, i, 1, *mtX, y, subsetX, subsetY, &nPositive, &positiveDataSize);

            /* Prepare "negative" observations of the training subset */
            size_t negativeDataSize;
            copyDataIntoSubtable(nFeatures, nVectors, j, -1, *mtX, y,
                                 subsetX + positiveDataSize, subsetY + nPositive, &nNegative, &negativeDataSize);
        }
        nTotal = nPositive + nNegative;
        subsetXTable->setNumberOfRows(nTotal);
        subsetYTable->setNumberOfRows(nTotal);

        /* Train "simple" classifier for pair of labels (i, j) */
        services::SharedPtr<classifier::training::Batch> simpleTraining = subsetLocal->simpleTraining;
        simpleTraining->input.set(classifier::training::data, subsetXTable);
        simpleTraining->input.set(classifier::training::labels, subsetYTable);
        simpleTraining->resetResult();

        int oldNumberOfThreads = fpk_serv_set_num_threads_local(1);
        simpleTraining->computeNoThrow();
        fpk_serv_set_num_threads_local(oldNumberOfThreads);
        if(simpleTraining->getErrors()->size() != 0)
        {
            subsetLocal->error.setId(services::ErrorMultiClassFailedToTrainTwoClassClassifier);
            return;
        }

        model->setTwoClassClassifierModel(imodel, simpleTraining->getResult()->get(classifier::training::model));
    } );

    subset.reduce([=](MultiClassClassifierTls<algorithmFPType, cpu> *subsetLocal)
    {
        if(subsetLocal->error.id() != services::NoErrorMessageFound)
        {
            this->_errors->add(services::SharedPtr<services::Error>(new services::Error(subsetLocal->error)));
        }
        delete subsetLocal;
    } );

    mtY.release();
}

template<typename algorithmFPType, CpuType cpu>
void MultiClassClassifierTrainKernel<oneAgainstOne, algorithmFPType, cpu>::
    computeDataSize(size_t nVectors, size_t nFeatures, size_t nClasses,
                    const NumericTable *xTable, int *y, size_t *nSubsetVectorsPtr, size_t *dataSizePtr)
{
    size_t *buffer = service_calloc<size_t, cpu>(4 * nClasses * sizeof(size_t));
    if (!buffer) { this->_errors->add(services::ErrorMemoryAllocationFailed); return; }
    size_t *classLabelsCount        = buffer;
    size_t *classNonZeroValuesCount = buffer + nClasses;
    size_t *classDataSize           = buffer + 2 * nClasses;
    size_t *classIndex              = buffer + 3 * nClasses;
    for (size_t i = 0; i < nVectors; i++)
    {
        classLabelsCount[y[i]]++;
    }
    if (xTable->getDataLayout() == NumericTableIface::csrArray)
    {
        CSRBlockMicroTable<algorithmFPType, readOnly, cpu> mtX(xTable);
        algorithmFPType *data;
        size_t *colIndices, *rowOffsets;
        mtX.getSparseBlock(0, nVectors, &data, &colIndices, &rowOffsets);

        /* Compute data size needed to store the largest subset of input tables */
        for (size_t i = 0; i < nVectors; i++)
        {
            classNonZeroValuesCount[y[i]] += (rowOffsets[i + 1] - rowOffsets[i]);
        }
        mtX.release();

        for (size_t i = 0; i < nClasses; i++)
        {
            classDataSize[i] = classLabelsCount[i] + classNonZeroValuesCount[i];
            classIndex[i] = i;
        }

        daal::algorithms::internal::qSort<size_t, size_t, cpu>(nClasses, classDataSize, classIndex);
        *nSubsetVectorsPtr = classLabelsCount       [classIndex[nClasses - 1]] + classLabelsCount       [classIndex[nClasses - 2]];
        *dataSizePtr       = classNonZeroValuesCount[classIndex[nClasses - 1]] + classNonZeroValuesCount[classIndex[nClasses - 2]];
    }
    else
    {
        daal::algorithms::internal::qSort<size_t, cpu>(nClasses, classLabelsCount);
        size_t nSubsetVectors = classLabelsCount[nClasses - 1] + classLabelsCount[nClasses - 2];
        *nSubsetVectorsPtr = nSubsetVectors;
        *dataSizePtr = nFeatures * nSubsetVectors;
    }
    daal::services::daal_free(buffer);
}

template<typename algorithmFPType, CpuType cpu>
void MultiClassClassifierTrainKernel<oneAgainstOne, algorithmFPType, cpu>::
    copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label,
                         BlockMicroTable  <algorithmFPType, readOnly, cpu> &mtX,
                         const int *y, algorithmFPType *subsetX, algorithmFPType *subsetY,
                         size_t *nRowsPtr, size_t *dataSize)
{
    size_t nRows = 0;
    algorithmFPType *xRow;
    for (size_t ix = 0; ix < nVectors; ix++)
    {
        if (y[ix] == classIdx)
        {
            mtX.getBlockOfRows(ix, 1, &xRow);
          PRAGMA_IVDEP
          PRAGMA_VECTOR_ALWAYS
            for (size_t jx = 0; jx < nFeatures; jx++)
            {
                subsetX[nRows * nFeatures + jx] = xRow[jx];
            }
            subsetY[nRows] = label;
            mtX.release();
            nRows++;
        }
    }
    *nRowsPtr = nRows;
    *dataSize = nRows * nFeatures;
}

template<typename algorithmFPType, CpuType cpu>
void MultiClassClassifierTrainKernel<oneAgainstOne, algorithmFPType, cpu>::
    copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label,
                         CSRBlockMicroTable<algorithmFPType, readOnly, cpu> &mtX,
                         const int *y, algorithmFPType *subsetX, size_t *colIndicesX, size_t *rowOffsetsX,
                         algorithmFPType *subsetY, size_t *nRowsPtr, size_t *dataSize)
{
    size_t nRows = 0;
    algorithmFPType *xRow;
    size_t *colIndices, *rowOffsets;
    size_t dataIndex = 0;
    for (size_t ix = 0; ix < nVectors; ix++)
    {
        if (y[ix] == classIdx)
        {
            mtX.getSparseBlock(ix, 1, &xRow, &colIndices, &rowOffsets);
            size_t nNonZeroValuesInRow = rowOffsets[1] - rowOffsets[0];
          PRAGMA_IVDEP
          PRAGMA_VECTOR_ALWAYS
            for (size_t jx = 0; jx < nNonZeroValuesInRow; jx++, dataIndex++)
            {
                subsetX[dataIndex] = xRow[jx];
                colIndicesX[dataIndex] = colIndices[jx];
            }
            rowOffsetsX[nRows + 1] = rowOffsetsX[nRows] + nNonZeroValuesInRow;
            subsetY[nRows] = label;
            mtX.release();
            nRows++;
        }
    }
    *nRowsPtr = nRows;
    *dataSize = rowOffsetsX[nRows] - rowOffsetsX[0];
}

} // namespace internal
} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif

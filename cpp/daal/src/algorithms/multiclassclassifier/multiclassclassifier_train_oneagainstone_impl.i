/* file: multiclassclassifier_train_oneagainstone_impl.i */
/*******************************************************************************
* Copyright 2014-2021 Intel Corporation
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

#include "algorithms/multi_class_classifier/multi_class_classifier_model.h"
#include "algorithms/svm/svm_model.h"

#include "src/threading/threading.h"
#include "src/algorithms/service_sort.h"
#include "src/algorithms/service_error_handling.h"
#include "src/externals/service_blas.h"

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
using namespace daal::internal;
using namespace daal::services::internal;
using namespace daal::data_management;
using namespace daal::services;
using namespace multi_class_classifier::internal;
using namespace svm::training::internal;

template <typename algorithmFPType, typename ClsType, CpuType cpu>
services::Status MultiClassClassifierTrainKernel<oneAgainstOne, algorithmFPType, ClsType, cpu>::compute(
    const NumericTable * xTable, const NumericTable * yTable, const NumericTable * wTable, daal::algorithms::Model * m, SvmModel * svmModel,
    const KernelParameter & par)
{
    Status s;

    const bool isOutSvmModel = svmModel != nullptr;
    Model * model            = static_cast<Model *>(m);

    const size_t nVectors = xTable->getNumberOfRows();
    ReadColumns<algorithmFPType, cpu> mtY(*const_cast<NumericTable *>(yTable), 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtY);
    const algorithmFPType * y = mtY.get();

    ReadColumns<algorithmFPType, cpu> mtW(const_cast<NumericTable *>(wTable), 0, 0, nVectors);
    DAAL_CHECK_BLOCK_STATUS(mtW);
    const algorithmFPType * weights = mtW.get();

    const size_t nFeatures = xTable->getNumberOfColumns();
    model->setNFeatures(nFeatures);
    services::SharedPtr<ClsType> simpleTrainingInit = par.training->clone();

    const size_t nClasses = par.nClasses;
    /* Compute data size needed to store the largest subset of input tables */
    size_t nSubsetVectors, dataSize;
    DAAL_CHECK_STATUS(s, computeDataSize(nVectors, nFeatures, nClasses, xTable, y, nSubsetVectors, dataSize));

    typedef SubTask<algorithmFPType, ClsType, cpu> TSubTask;
    /* Allocate memory for storing subsets of input data */
    daal::ls<TSubTask *> lsTask([=, &simpleTrainingInit]() {
        if (xTable->getDataLayout() == NumericTableIface::csrArray)
            return (TSubTask *)SubTaskCSR<algorithmFPType, ClsType, cpu>::create(nFeatures, nSubsetVectors, dataSize, xTable, weights,
                                                                                 simpleTrainingInit);
        return (TSubTask *)SubTaskDense<algorithmFPType, ClsType, cpu>::create(nFeatures, nSubsetVectors, dataSize, xTable, weights,
                                                                               simpleTrainingInit);
    });

    SafeStatus safeStat;

    TArray<bool, cpu> is_sv(nVectors);
    bool * const data_is_sv = is_sv.get();
    daal::services::internal::service_memset_seq<bool, cpu>(data_is_sv, false, nVectors);

    DAAL_CHECK_MALLOC(is_sv.get());

    const size_t nModels = (nClasses * (nClasses - 1)) >> 1;
    // daal::threader_for(nModels, nModels, [&](size_t imodel) {
    size_t imodel = 0;
    for (size_t i = 0; i < nClasses; i++)
    {
        for (size_t j = i + 1; j < nClasses; j++)
        {
            /* Find indices of positive and negative classes for current model */
            // size_t i    = 1; /* index of the positive class */
            // size_t j    = 0; /* index of the negative class */
            // size_t isum = 0;  isum = (i*i - i) / 2;
            // while (i <= j || isum + j != imodel)
            // {
            //     isum += i;
            //     i++;
            //     j = imodel - isum;
            // }

            TSubTask * local = lsTask.local();
            // if (!local)
            // {
            //     safeStat.add(services::ErrorMemoryAllocationFailed);
            //     return;
            // }
            DAAL_LS_RELEASE(TSubTask, lsTask, local); //releases local storage when leaving this scope

            size_t nRowsInSubset = 0;
            local->getDataSubset(nFeatures, nVectors, i, j, y, nRowsInSubset);
            // DAAL_CHECK_STATUS_THR(s);
            classifier::ModelPtr pModel;
            if (nRowsInSubset)
            {
                s = local->trainSimpleClassifier(nRowsInSubset);
                // if (!s)
                // {
                //     safeStat |= s;
                //     safeStat.add(services::ErrorMultiClassFailedToTrainTwoClassClassifier);
                //     return;
                // }
                pModel = local->getModel();
            }
            model->setTwoClassClassifierModel(imodel, pModel);
            if (svmModel)

            {
                auto svm_model            = daal::services::staticPointerCast<svm::Model>(pModel);
                auto two_class_sv_ind     = svm_model->getSupportIndices();
                const size_t count_sv_ind = two_class_sv_ind->getNumberOfRows();

                ReadColumns<int, cpu> mtSvIndex(two_class_sv_ind.get(), 0, 0, count_sv_ind);
                DAAL_CHECK_BLOCK_STATUS(mtY);
                const int * arr_two_class_sv_ind = mtSvIndex.get();

                for (std::int64_t svId = 0; svId < count_sv_ind; ++svId)
                {
                    const std::int64_t original_index = arr_two_class_sv_ind[svId];
                    data_is_sv[original_index]        = true;
                }
                auto biasesTable = svmModel->getBiases();
                WriteOnlyColumns<algorithmFPType, cpu> mtBiases(biasesTable.get(), 0, imodel, 1);
                *mtBiases.get() = svm_model->getBias();
            }

            ++imodel;
        }
    }
    // });
    lsTask.reduce([=, &safeStat](TSubTask * local) { delete local; });
    if (svmModel)
    {
        TArray<size_t, cpu> array_sv_counts(nClasses);
        size_t * const sv_counts = array_sv_counts.get();
        size_t nSV               = 0;

        for (size_t iClass = 0; iClass < nClasses; ++iClass)
        {
            sv_counts[iClass] = 0;
            for (size_t j = 0; j < nVectors; ++j)
            {
                const size_t label = size_t(y[j]);
                if (data_is_sv[j] && label == iClass)
                {
                    ++sv_counts[iClass];
                }
            }
            nSV += sv_counts[iClass];
            printf("%lu: %lu\n", iClass, sv_counts[iClass]);
        }
        printf("sumSV: %lu\n", nSV);
        NumericTablePtr supportIndicesTable = svmModel->getSupportIndices();
        DAAL_CHECK_STATUS(s, supportIndicesTable->resize(nSV));

        TArray<size_t, cpu> svGroupByClassArray(nSV);
        size_t * const svGroupByClass = svGroupByClassArray.get();

        TArray<size_t, cpu> svIndMappingArray(nVectors);
        size_t * const svIndMapping = svIndMappingArray.get();
        {
            WriteOnlyColumns<int, cpu> mtSupportIndices(supportIndicesTable.get(), 0, 0, nSV);
            int * supportIndices = mtSupportIndices.get();

            size_t inxSV = 0;
            for (size_t iClass = 0; iClass < nClasses; ++iClass)
            {
                for (size_t j = 0; j < nVectors; ++j)
                {
                    const size_t label = static_cast<size_t>(y[j]);
                    if (data_is_sv[j] && label == iClass)
                    {
                        supportIndices[inxSV] = j;
                        svIndMapping[j]       = inxSV;
                        inxSV++;
                    }
                }
            }
        }

        NumericTablePtr coeffOutTable = svmModel->getCoefficients();
        DAAL_CHECK_STATUS(s, coeffOutTable->resize(nSV));

        using SvmResultTask = SaveResultTask<algorithmFPType, cpu>;
        DAAL_CHECK_STATUS(s, SvmResultTask::setSVByIndices(NumericTablePtr(const_cast<NumericTable *>(xTable)), supportIndicesTable,
                                                           svmModel->getSupportVectors()));
        WriteOnlyRows<algorithmFPType, cpu> mtCoefficientsOut(coeffOutTable.get(), 0, coeffOutTable->getNumberOfRows());
        DAAL_CHECK_BLOCK_STATUS(mtCoefficientsOut);
        algorithmFPType * const coefficientsOut = mtCoefficientsOut.get();
        daal::services::internal::service_memset<algorithmFPType, cpu>(coefficientsOut, algorithmFPType(0),
                                                                       coeffOutTable->getNumberOfRows() * coeffOutTable->getNumberOfColumns());
        imodel = 0;
        for (size_t firstClass = 0; firstClass < nClasses; ++firstClass)
        {
            for (size_t twoClass = firstClass + 1; twoClass < nClasses; ++twoClass)
            {
                auto svmModelId   = daal::services::staticPointerCast<svm::Model>(model->getTwoClassClassifierModel(imodel));
                auto coeffInTable = svmModelId->getClassificationCoefficients();
                ReadRows<algorithmFPType, cpu> mtCoefficientsIn(coeffInTable.get(), 0, coeffInTable->getNumberOfRows());
                DAAL_CHECK_BLOCK_STATUS(mtCoefficientsIn);
                const algorithmFPType * const coeffIn = mtCoefficientsIn.get();

                auto twoClassSvIndTable = svmModelId->getSupportIndices();
                const size_t svCounts   = twoClassSvIndTable->getNumberOfRows();
                ReadColumns<int, cpu> mtSvIndex(twoClassSvIndTable.get(), 0, 0, svCounts);
                DAAL_CHECK_BLOCK_STATUS(mtSvIndex);
                const int * twoClassSvInd = mtSvIndex.get();

                for (size_t sv = 0; sv < svCounts; ++sv)
                {
                    const size_t originalInd = twoClassSvInd[sv];
                    const size_t label       = static_cast<size_t>(y[originalInd]);
                    const size_t colInd      = svIndMapping[originalInd];
                    if (label == firstClass)
                    {
                        coefficientsOut[(twoClass - 1) * nSV + colInd] = coeffIn[sv];
                    }
                    else if (label == twoClass)
                    {
                        coefficientsOut[(firstClass)*nSV + colInd] = coeffIn[sv];
                    }
                }
            }
            ++imodel;
        }
    }

    return safeStat.detach();
}

template <typename algorithmFPType, typename ClsType, CpuType cpu>
Status MultiClassClassifierTrainKernel<oneAgainstOne, algorithmFPType, ClsType, cpu>::computeDataSize(size_t nVectors, size_t nFeatures,
                                                                                                      size_t nClasses, const NumericTable * xTable,
                                                                                                      const algorithmFPType * y,
                                                                                                      size_t & nSubsetVectors, size_t & dataSize)
{
    TArray<size_t, cpu> buffer(4 * nClasses);
    DAAL_CHECK_MALLOC(buffer.get());
    daal::services::internal::service_memset_seq<size_t, cpu>(buffer.get(), 0, buffer.size());
    size_t * classLabelsCount        = buffer.get();
    size_t * classNonZeroValuesCount = buffer.get() + nClasses;
    size_t * classDataSize           = buffer.get() + 2 * nClasses;
    size_t * classIndex              = buffer.get() + 3 * nClasses;
    for (size_t i = 0; i < nVectors; ++i)
    {
        ++classLabelsCount[size_t(y[i])];
    }
    if (xTable->getDataLayout() == NumericTableIface::csrArray)
    {
        CSRNumericTableIface * csrIface = dynamic_cast<CSRNumericTableIface *>(const_cast<NumericTable *>(xTable));
        ReadRowsCSR<algorithmFPType, cpu> _mtX(*csrIface, 0, nVectors);
        DAAL_CHECK_BLOCK_STATUS(_mtX);
        const size_t * rowOffsets = _mtX.rows();
        /* Compute data size needed to store the largest subset of input tables */
        for (size_t i = 0; i < nVectors; ++i)
        {
            classNonZeroValuesCount[size_t(y[i])] += (rowOffsets[i + 1] - rowOffsets[i]);
        }

        for (size_t i = 0; i < nClasses; ++i)
        {
            classDataSize[i] = classLabelsCount[i] + classNonZeroValuesCount[i];
            classIndex[i]    = i;
        }

        daal::algorithms::internal::qSort<size_t, size_t, cpu>(nClasses, classDataSize, classIndex);
        const auto idx1 = classIndex[nClasses - 1], idx2 = classIndex[nClasses - 2];
        dataSize = classNonZeroValuesCount[idx1] + classNonZeroValuesCount[idx2];
        daal::algorithms::internal::qSort<size_t, cpu>(nClasses, classLabelsCount);
        nSubsetVectors = classLabelsCount[nClasses - 1] + classLabelsCount[nClasses - 2];
    }
    else
    {
        daal::algorithms::internal::qSort<size_t, cpu>(nClasses, classLabelsCount);
        nSubsetVectors = classLabelsCount[nClasses - 1] + classLabelsCount[nClasses - 2];
        dataSize       = nFeatures * nSubsetVectors;
    }

    return Status();
}

template <typename algorithmFPType, typename ClsType, CpuType cpu>
Status SubTaskDense<algorithmFPType, ClsType, cpu>::copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label,
                                                                         const algorithmFPType * y, size_t & nRows)
{
    for (size_t ix = 0; ix < nVectors; ix++)
    {
        if (size_t(y[ix]) != classIdx) continue;
        _mtX.next(ix, 1);
        DAAL_CHECK_BLOCK_STATUS(_mtX);
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t jx = 0; jx < nFeatures; jx++) this->_subsetX.get()[nRows * nFeatures + jx] = _mtX.get()[jx];
        this->_subsetY[nRows] = label;
        if (this->_weights)
        {
            this->_subsetW[nRows] = this->_weights[ix];
        }
        ++nRows;
    }
    return Status();
}

template <typename algorithmFPType, typename ClsType, CpuType cpu>
Status SubTaskCSR<algorithmFPType, ClsType, cpu>::copyDataIntoSubtable(size_t nFeatures, size_t nVectors, int classIdx, algorithmFPType label,
                                                                       const algorithmFPType * y, size_t & nRows)
{
    _rowOffsetsX[0]  = 1;
    size_t dataIndex = (nRows ? _rowOffsetsX[nRows] - _rowOffsetsX[0] : 0);
    for (size_t ix = 0; ix < nVectors; ix++)
    {
        if (size_t(y[ix]) != classIdx) continue;
        _mtX.next(ix, 1);
        DAAL_CHECK_BLOCK_STATUS(_mtX);
        const size_t nNonZeroValuesInRow = _mtX.rows()[1] - _mtX.rows()[0];
        const size_t * colIndices        = _mtX.cols();
        PRAGMA_IVDEP
        PRAGMA_VECTOR_ALWAYS
        for (size_t jx = 0; jx < nNonZeroValuesInRow; ++jx, ++dataIndex)
        {
            this->_subsetX.get()[dataIndex] = _mtX.values()[jx];
            _colIndicesX[dataIndex]         = colIndices[jx];
        }
        _rowOffsetsX[nRows + 1] = _rowOffsetsX[nRows] + nNonZeroValuesInRow;
        this->_subsetY[nRows]   = label;
        if (this->_weights)
        {
            this->_subsetW[nRows] = this->_weights[ix];
        }

        ++nRows;
    }
    return Status();
}

} // namespace internal
} // namespace training
} // namespace multi_class_classifier
} // namespace algorithms
} // namespace daal

#endif

/* file: svm_train_workset_oneapi.h */
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
//  SVM workset structure implementation
//--
*/

#ifndef __SVM_TRAIN_WORKSET_ONEAPI_H__
#define __SVM_TRAIN_WORKSET_ONEAPI_H__

#include "service/kernel/service_utils.h"
#include "algorithms/kernel/svm/oneapi/svm_helper_oneapi.h"

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
using namespace daal::services::internal;

template <typename algorithmFPType>
struct TaskWorkingSet
{
    using Helper = utils::internal::HelperSVM<algorithmFPType>;

    TaskWorkingSet(const size_t nVectors) : _nVectors(nVectors) {}

    services::Status init()
    {
        services::Status status;
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

        _sortedFIndices = context.allocate(TypeIds::id<uint32_t>(), _nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);

        _indicator = context.allocate(TypeIds::id<uint32_t>(), _nVectors, &status);
        context.fill(_indicator, 0, &status);
        DAAL_CHECK_STATUS_VAR(status);

        auto & deviceInfo = context.getInfoDevice();

        const size_t maxWS = deviceInfo.maxWorkGroupSize;

        _nWS       = utils::internal::min(utils::internal::maxpow2(_nVectors), utils::internal::maxpow2(maxWS));
        _nSelected = 0;

        _valuesSort     = context.allocate(TypeIds::id<algorithmFPType>(), _nVectors, &status);
        _valuesSortBuff = context.allocate(TypeIds::id<algorithmFPType>(), _nVectors, &status);
        _buffIndices    = context.allocate(TypeIds::id<uint32_t>(), _nVectors, &status);

        _wsIndices = context.allocate(TypeIds::id<uint32_t>(), _nWS, &status);
        return status;
    }

    size_t getSize() const { return _nWS; }

    services::Status copyLastToFirst()
    {
        const size_t q = _nWS / 2;
        services::Status status;
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        context.copy(_wsIndices, 0, _wsIndices, q, _nWS - q, &status);
        _nSelected = q;
        return status;
    }

    services::Status selectWS(const services::Buffer<algorithmFPType> & yBuff, const services::Buffer<algorithmFPType> & alphaBuff,
                              const services::Buffer<algorithmFPType> & fBuff, const algorithmFPType C)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(selectWS);
        services::Status status;
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

        auto wsIndicesBuff = _wsIndices.get<uint32_t>();
        auto indicatorBuff = _indicator.get<uint32_t>();

        DAAL_CHECK_STATUS(status, Helper::argSort(fBuff, _valuesSort, _valuesSortBuff, _sortedFIndices, _buffIndices, _nVectors));

        DAAL_CHECK_STATUS_VAR(status);

        {
            const size_t nNeedSelect = (_nWS - _nSelected) / 2;

            DAAL_CHECK_STATUS(status, Helper::checkUpper(yBuff, alphaBuff, indicatorBuff, C, _nVectors));

            /* Reset indicator for busy indeces */
            if (_nSelected > 0)
            {
                DAAL_CHECK_STATUS(status, resetIndicatorWithZeros(wsIndicesBuff, indicatorBuff, _nSelected));
            }

            size_t nUpperSelect = 0;
            DAAL_CHECK_STATUS(status, Partition::flaggedIndex(indicatorBuff, _sortedFIndices, _buffIndices, _nVectors, nUpperSelect));

            const size_t nCopy = utils::internal::min(nUpperSelect, nNeedSelect);

            context.copy(_wsIndices, _nSelected, _buffIndices, 0, nCopy, &status);
            _nSelected += nCopy;
        }

        {
            const size_t nNeedSelect = _nWS - _nSelected;

            DAAL_CHECK_STATUS(status, Helper::checkLower(yBuff, alphaBuff, indicatorBuff, C, _nVectors));

            /* Reset indicator for busy indeces */
            if (_nSelected > 0)
            {
                DAAL_CHECK_STATUS(status, resetIndicatorWithZeros(wsIndicesBuff, indicatorBuff, _nSelected));
            }

            size_t nLowerSelect = 0;
            DAAL_CHECK_STATUS(status, Partition::flaggedIndex(indicatorBuff, _sortedFIndices, _buffIndices, _nVectors, nLowerSelect));

            const size_t nCopy = utils::internal::min(nLowerSelect, nNeedSelect);

            /* Copy latest nCopy elements */
            context.copy(_wsIndices, _nSelected, _buffIndices, nLowerSelect - nCopy, nCopy, &status);
            _nSelected += nCopy;
        }

        if (_nSelected < _nWS)
        {
            const size_t nNeedSelect = _nWS - _nSelected;

            DAAL_CHECK_STATUS(status, Helper::checkUpper(yBuff, alphaBuff, indicatorBuff, C, _nVectors));

            /* Reset indicator for busy indeces */
            if (_nSelected > 0)
            {
                DAAL_CHECK_STATUS(status, resetIndicatorWithZeros(wsIndicesBuff, indicatorBuff, _nSelected));
            }

            size_t nUpperSelect = 0;
            DAAL_CHECK_STATUS(status, Partition::flaggedIndex(indicatorBuff, _sortedFIndices, _buffIndices, _nVectors, nUpperSelect));

            const size_t nCopy = utils::internal::min(nUpperSelect, nNeedSelect);

            context.copy(_wsIndices, _nSelected, _buffIndices, 0, nCopy, &status);
            _nSelected += nCopy;
        }

        DAAL_ASSERT(_nSelected == _nWS);

        _nSelected = 0;
        return status;
    }

    const services::Buffer<uint32_t> & getWSIndeces() const { return _wsIndices.get<uint32_t>(); }

    services::Status resetIndicatorWithZeros(const services::Buffer<uint32_t> & idx, services::Buffer<uint32_t> & indicator, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(resetIndicatorWithZeros);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = Helper::buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("resetIndicatorWithZeros");

        KernelArguments args(2);
        args.set(0, idx, AccessModeIds::read);
        args.set(1, indicator, AccessModeIds::write);

        KernelRange range(n);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

private:
    size_t _nSelected;
    size_t _nVectors;
    size_t _nWS;

    UniversalBuffer _sortedFIndices;
    UniversalBuffer _indicator;
    UniversalBuffer _wsIndices;
    UniversalBuffer _buffIndices;
    UniversalBuffer _valuesSort;
    UniversalBuffer _valuesSortBuff;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

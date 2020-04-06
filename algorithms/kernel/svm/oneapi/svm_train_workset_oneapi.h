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
struct TaskWorkingSet
{
    using Helper = HelperSVM<algorithmFPType>;

    TaskWorkingSet(const size_t nVectors, const bool verbose) : _nVectors(nVectors), _verbose(verbose) {}

    services::Status init()
    {
        services::Status status;
        auto & context    = services::Environment::getInstance()->getDefaultExecutionContext();
        const auto idType = TypeIds::id<algorithmFPType>();

        _sortedFIndices = context.allocate(TypeIds::id<int>(), _nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);

        _indicator = context.allocate(TypeIds::id<int>(), _nVectors, &status);
        context.fill(_indicator, 0, &status);
        DAAL_CHECK_STATUS_VAR(status);

        auto & deviceInfo = context.getInfoDevice();

        const size_t maxWS = deviceInfo.max_work_group_size;

        _nWS       = min(maxpow2(_nVectors), maxWS);
        _nSelected = 0;

        _valuesSort  = context.allocate(TypeIds::id<algorithmFPType>(), _nVectors, &status);
        _buffIndices = context.allocate(TypeIds::id<int>(), _nVectors, &status);

        _wsIndices     = context.allocate(TypeIds::id<int>(), _nWS, &status);
        _wsSaveIndices = context.allocate(TypeIds::id<int>(), _nWS, &status);
        return status;
    }

    size_t getSize() const { return _nWS; }

    services::Status saveWSIndeces()
    {
        const size_t q = _nWS / 2;
        services::Status status;
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        // context.copy(_wsIndices, 0, _wsIndices, q, _nWS - q, &status);
        context.copy(_wsIndices, 0, _wsSaveIndices, q, _nWS - q, &status);
        _nSelected = q;
        return status;
    }

    services::Status selectWS(const services::Buffer<algorithmFPType> & yBuff, const services::Buffer<algorithmFPType> & alphaBuff,
                              const services::Buffer<algorithmFPType> & fBuff, const algorithmFPType C)
    {
        services::Status status;
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

        if (_verbose)
        {
            printf(">>>> selectWS\n");
        }

        context.fill(_buffIndices, 0, &status);
        context.fill(_indicator, 0, &status);

        auto sortedFIndicesBuff = _sortedFIndices.get<int>();
        auto tmpIndicesBuff     = _buffIndices.get<int>();
        auto wsIndicesBuff      = _wsIndices.get<int>();
        auto indicatorBuff      = _indicator.get<int>();

        DAAL_CHECK_STATUS(status, Helper::argSort(fBuff, _valuesSort, _sortedFIndices, _buffIndices, _nVectors));

        if (_verbose)
        {
            printf(">> argSort: ");
            {
                int * sortedFIndices_host = sortedFIndicesBuff.toHost(ReadWriteMode::readOnly).get();
                for (int i = 0; i < min(16ul, _nWS); i++)
                {
                    printf("%d ", sortedFIndices_host[i]);
                }
                printf(" ... ");
                for (int i = _nVectors - 1; i >= _nVectors - min(16ul, _nWS); i--)
                {
                    printf("%d ", sortedFIndices_host[i]);
                }
            }
            printf("\n");
            printf(">> sort val: ");
            {
                int * sortedFIndices_host = sortedFIndicesBuff.toHost(ReadWriteMode::readOnly).get();
                algorithmFPType * f_host  = fBuff.toHost(ReadWriteMode::readOnly).get();
                for (int i = 0; i < min(16ul, _nWS); i++)
                {
                    printf("%.2f ", f_host[sortedFIndices_host[i]]);
                }
                printf(" ... ");
                for (int i = _nVectors - 1; i >= _nVectors - min(16ul, _nWS); i--)
                {
                    printf("%.2f ", f_host[sortedFIndices_host[i]]);
                }
            }
            printf("\n");
        }

        DAAL_CHECK_STATUS_VAR(status);

        {
            const size_t nNeedSelect = (_nWS - _nSelected) / 2;

            DAAL_CHECK_STATUS(status, Helper::checkUpper(yBuff, alphaBuff, indicatorBuff, C, _nVectors));

            /* Reset indicator for busy indeces */
            if (_nSelected > 0)
            {
                DAAL_CHECK_STATUS(status, resetIndecator(wsIndicesBuff, indicatorBuff, _nSelected));
            }

            size_t nUpperSelect = 0;
            DAAL_CHECK_STATUS(status, Helper::gatherIndices(indicatorBuff, sortedFIndicesBuff, _nVectors, tmpIndicesBuff, nUpperSelect));

            const size_t nCopy = min(nUpperSelect, nNeedSelect);

            context.copy(_wsIndices, _nSelected, _buffIndices, 0, nCopy, &status);

            _nSelected += nCopy;

            if (_verbose)
            {
                printf(">> CheckUpper[_buffIndices] - selectUpper:%lu _nSelected: %lu \n", nUpperSelect, _nSelected);
            }
        }

        {
            const size_t nNeedSelect = _nWS - _nSelected;

            DAAL_CHECK_STATUS(status, Helper::checkLower(yBuff, alphaBuff, indicatorBuff, C, _nVectors));

            /* Reset indicator for busy indeces */
            if (_nSelected > 0)
            {
                DAAL_CHECK_STATUS(status, resetIndecator(wsIndicesBuff, indicatorBuff, _nSelected));
            }

            size_t nLowerSelect = 0;
            DAAL_CHECK_STATUS(status, Helper::gatherIndices(indicatorBuff, sortedFIndicesBuff, _nVectors, tmpIndicesBuff, nLowerSelect));

            const size_t nCopy = min(nLowerSelect, nNeedSelect);

            /* Copy latest nCopy elements */
            context.copy(_wsIndices, _nSelected, _buffIndices, nLowerSelect - nCopy, nCopy, &status);

            _nSelected += nCopy;

            if (_verbose)
            {
                printf(">> checkLower[_buffIndices] - selectLower: %lu _nSelected: %lu \n", nLowerSelect, _nSelected);
            }
        }

        if (_nSelected < _nWS)
        {
            const size_t nNeedSelect = _nWS - _nSelected;

            DAAL_CHECK_STATUS(status, Helper::checkUpper(yBuff, alphaBuff, indicatorBuff, C, _nVectors));

            /* Reset indicator for busy indeces */
            if (_nSelected > 0)
            {
                DAAL_CHECK_STATUS(status, resetIndecator(wsIndicesBuff, indicatorBuff, _nSelected));
            }

            size_t nUpperSelect = 0;
            DAAL_CHECK_STATUS(status, Helper::gatherIndices(indicatorBuff, sortedFIndicesBuff, _nVectors, tmpIndicesBuff, nUpperSelect));

            const size_t nCopy = min(nUpperSelect, nNeedSelect);

            context.copy(_wsIndices, _nSelected, _buffIndices, 0, nCopy, &status);

            _nSelected += nCopy;

            if (_verbose)
            {
                printf("!!! _nSelected < _nWS - %lu %lu: \n", _nSelected, _nWS);
            }
        }

        DAAL_ASSERT(_nSelected == _nWS);

        if (_verbose)
        {
            printf(">> _wsIndices:  ");
            {
                int * wsIndexes_host = _wsIndices.get<int>().toHost(ReadWriteMode::readOnly).get();
                for (int i = 0; i < min(16ul, _nWS); i++)
                {
                    printf("%d ", wsIndexes_host[i]);
                }
            }
            printf("\n");
        }

        context.copy(_wsSaveIndices, 0, _wsIndices, 0, _nWS, &status);
        _nSelected = 0;
        return status;
    }

    const services::Buffer<int> & getSortedFIndices() const { return _sortedFIndices.get<int>(); }
    const services::Buffer<int> & getWSIndeces() const { return _wsIndices.get<int>(); }

    services::Status resetIndecator(const services::Buffer<int> & idx, services::Buffer<int> & indicator, const size_t n)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(resetIndecator);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = Helper::buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("resetIndecator");

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

    bool _verbose;

    UniversalBuffer _sortedFIndices;
    UniversalBuffer _indicator;
    UniversalBuffer _wsIndices;
    UniversalBuffer _wsSaveIndices;
    UniversalBuffer _buffIndices;
    UniversalBuffer _valuesSort;
};

} // namespace internal
} // namespace training
} // namespace svm
} // namespace algorithms
} // namespace daal

#endif

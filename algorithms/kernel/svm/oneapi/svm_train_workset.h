/* file: svm_train_cache.h */
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
//  SVM cache structure implementation
//--
*/

#ifndef __SVM_TRAIN_WORKSET_H__
#define __SVM_TRAIN_WORKSET_H__

#include "service/kernel/service_utils.h"
#include "algorithms/kernel/svm/oneapi/svm_helper.h"

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

        sortedFIndices = context.allocate(TypeIds::id<int>(), _nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);

        indicator = context.allocate(TypeIds::id<int>(), _nVectors, &status);
        context.fill(indicator, 0.0, &status);
        DAAL_CHECK_STATUS_VAR(status);

        fIndices = context.allocate(TypeIds::id<int>(), _nVectors, &status);
        DAAL_CHECK_STATUS_VAR(status);
        auto fIndicesBuf = fIndices.get<int>();
        DAAL_CHECK_STATUS(status, Helper::range(fIndicesBuf, _nVectors));

        // TODO: Get from device info
        const size_t maxWS = 16;
        _nWS                = min(maxWS, _nVectors);

        tmpValues  = context.allocate(TypeIds::id<algorithmFPType>(), _nVectors, &status);
        wsIndices  = context.allocate(TypeIds::id<int>(), _nWS, &status);
        tmpIndices = context.allocate(TypeIds::id<int>(), _nVectors, &status);
        return status;
    }

    size_t getSize() const { return _nWS; }

    services::Status selectWS(const services::Buffer<algorithmFPType> & yBuff, const services::Buffer<algorithmFPType> & alphaBuff,
                              const services::Buffer<algorithmFPType> & fBuff, const algorithmFPType C)
    {
        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();

        if (_verbose)
        {
            printf(">>>> selectWS\n");
        }

        services::Status status;
        static bool firstCall = true;

        size_t nSelected = 0;

        if (firstCall)
        {
            firstCall = false;
        }
        else
        {
            // TODO!
            nSelected = 1;
            // copy
        }
        const size_t q = _nWS - nSelected;


        auto sortedFIndicesBuff = sortedFIndices.get<int>();
        auto tmpIndicesBuff = tmpIndices.get<int>();
        auto fIndicesBuf = fIndices.get<int>();
        DAAL_CHECK_STATUS(status, Helper::argSort(fBuff, fIndices, tmpValues, sortedFIndices, _nVectors));

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
                    printf("%.2f ",  f_host[sortedFIndices_host[i]]);
                }

            }
            printf("\n");
        }

        context.fill(indicator, 0.0, &status);
        DAAL_CHECK_STATUS_VAR(status);
        auto indicatorBuff = indicator.get<int>();

        {
            const size_t nSelect = q / 2;

            DAAL_CHECK_STATUS(status, checkUpper(yBuff, alphaBuff, indicatorBuff, C, _nVectors));

            size_t selectUpper = 0;
            DAAL_CHECK_STATUS(status, Helper::gatherIndices(indicatorBuff, sortedFIndicesBuff, _nVectors, tmpIndicesBuff, selectUpper));

            if (_verbose)
            {
                printf(">> CheckUpper[tmpIndices] - %lu:  ", selectUpper);
                {
                    int * wsIndexes_host = tmpIndices.get<int>().toHost(ReadWriteMode::readOnly).get();
                    for (int i = 0; i < min(16ul, _nWS); i++)
                    {
                        printf("%d ", wsIndexes_host[i]);
                    }
                }
                printf("\n");
            }

            const size_t nCopy = min(selectUpper, nSelect);

            context.copy(wsIndices, nSelected, tmpIndices, 0, nCopy, &status);

            nSelected += nCopy;
        }

        {
            const size_t nSelect = _nWS - nSelected;

            DAAL_CHECK_STATUS(status, checkLower(yBuff, alphaBuff, indicatorBuff, C, _nVectors));

            size_t selectLower = 0;
            DAAL_CHECK_STATUS(status, Helper::gatherIndices(indicatorBuff, sortedFIndicesBuff, _nVectors, tmpIndicesBuff, selectLower));

            if (_verbose)
            {
                printf(">> checkLower[tmpIndices] - %lu:  ", selectLower);
                {
                    int * wsIndexes_host = tmpIndices.get<int>().toHost(ReadWriteMode::readOnly).get();
                    for (int i = 0; i < min(16ul, _nWS); i++)
                    {
                        printf("%d ", wsIndexes_host[i]);
                    }
                }
                printf("\n");
            }

            const size_t nCopy = min(selectLower, nSelect);

            // Get latest elements
            context.copy(wsIndices, nSelected, tmpIndices, selectLower - nCopy, nCopy, &status);

            nSelected += selectLower;
        }

        if (_verbose)
        {
            printf(">> wsIndices:  ");
            {
                int * wsIndexes_host = wsIndices.get<int>().toHost(ReadWriteMode::readOnly).get();
                for (int i = 0; i < min(16ul, _nWS); i++)
                {
                    printf("%d ", wsIndexes_host[i]);
                }
            }
            printf("\n");
        }

        return status;
    }

    const services::Buffer<int> & getSortedFIndices() const { return sortedFIndices.get<int>(); }
    const services::Buffer<int> & getWSIndeces() const { return wsIndices.get<int>(); }

    services::Status checkUpper(const services::Buffer<algorithmFPType> & yBuff, const services::Buffer<algorithmFPType> & alphaBuff,
                                services::Buffer<int> & indicatorBuff, const algorithmFPType C, const size_t nSelect)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkUpper);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = Helper::buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkUpper");

        KernelArguments args(4);
        args.set(0, yBuff, AccessModeIds::read);
        args.set(1, alphaBuff, AccessModeIds::read);
        args.set(2, C);
        args.set(3, indicatorBuff, AccessModeIds::readwrite);

        KernelRange range(nSelect);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

    services::Status checkLower(const services::Buffer<algorithmFPType> & yBuff, const services::Buffer<algorithmFPType> & alphaBuff,
                                services::Buffer<int> & indicatorBuff, const algorithmFPType C, const size_t nSelect)
    {
        DAAL_ITTNOTIFY_SCOPED_TASK(checkLower);

        auto & context = services::Environment::getInstance()->getDefaultExecutionContext();
        auto & factory = context.getClKernelFactory();

        services::Status status = Helper::buildProgram(factory);
        DAAL_CHECK_STATUS_VAR(status);

        auto kernel = factory.getKernel("checkLower");

        KernelArguments args(4);
        args.set(0, yBuff, AccessModeIds::read);
        args.set(1, alphaBuff, AccessModeIds::read);
        args.set(2, C);
        args.set(3, indicatorBuff, AccessModeIds::readwrite);

        KernelRange range(nSelect);

        context.run(range, kernel, args, &status);
        DAAL_CHECK_STATUS_VAR(status);

        return status;
    }

private:
    size_t _nVectors;
    size_t _nWS;

    bool _verbose;

    UniversalBuffer sortedFIndices;
    UniversalBuffer indicator;
    UniversalBuffer fIndices;
    UniversalBuffer wsIndices;
    UniversalBuffer tmpIndices;
    UniversalBuffer tmpValues;
};

} // namespace internal

} // namespace training

} // namespace svm

} // namespace algorithms

} // namespace daal

#endif

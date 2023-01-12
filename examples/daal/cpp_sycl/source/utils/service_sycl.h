/* file: service_sycl.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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
!  Content:
!    Auxiliary sycl functions used in C++ examples
!******************************************************************************/

#ifndef _SERVICE_SYCL_H
#define _SERVICE_SYCL_H

#include <list>
#include <memory>

#include <CL/cl.h>
#include <sycl/sycl.hpp>

#include "service.h"

std::unique_ptr<sycl::device> makeDevice(int (*selector)(const sycl::device&)) {
    try {
        return std::unique_ptr<sycl::device>(new sycl::device(selector));
    }
    catch (...) {
        return std::unique_ptr<sycl::device>();
    }
}

std::list<std::pair<std::string, sycl::device> > getListOfDevices() {
    std::list<std::pair<std::string, sycl::device> > selects;
    std::unique_ptr<sycl::device> device;

    device = makeDevice(&sycl::gpu_selector_v);
    if (device)
        selects.emplace_back("GPU", *device);

    device = makeDevice(&sycl::cpu_selector_v);
    if (device)
        selects.emplace_back("CPU", *device);

    return selects;
}

template <typename DataType>
daal::data_management::SyclCSRNumericTablePtr createSyclSparseTable(
    const std::string& datasetFileName) {
    auto numericTable = createSparseTable<DataType>(datasetFileName);

    DataType* data = nullptr;
    size_t* colIndices = nullptr;
    size_t* rowOffsets = nullptr;
    numericTable->getArrays(&data, &colIndices, &rowOffsets);

    auto dataBuff = sycl::buffer<DataType, 1>(data, numericTable->getDataSize());
    auto colIndicesBuff = sycl::buffer<size_t, 1>(colIndices, numericTable->getDataSize());
    auto rowOffsetsBuff = sycl::buffer<size_t, 1>(rowOffsets, numericTable->getNumberOfRows() + 1);

    auto syclNumericTable = daal::data_management::SyclCSRNumericTable::create<DataType>(
        dataBuff,
        colIndicesBuff,
        rowOffsetsBuff,
        numericTable->getNumberOfColumns(),
        numericTable->getNumberOfRows());

    return syclNumericTable;
}

#endif

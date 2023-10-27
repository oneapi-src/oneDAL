/** file algorithm_hyperparameter.cpp */
/*******************************************************************************
* Copyright 2023 Intel Corporation
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
//
//--
*/

#include "algorithms/algorithm_types.h"
#include "services/error_handling.h"
#include "src/algorithms/service_hash_table.h"

namespace daal
{
namespace algorithms
{
namespace internal
{
/**
 * \brief Implementation of the common for all algorithms performance-related hyperparameters of the computation.
 */
struct HyperparameterImpl : public HyperparameterBaseImpl
{
    DAAL_NEW_DELETE();

    HyperparameterImpl(size_t intParamCount, size_t doubleParamCount) : _iHT(intParamCount), _dHT(doubleParamCount) {}

    services::Status set(unsigned int id, int64_t value)
    {
        _iHT.insert(id, value);
        return services::Status();
    }

    services::Status set(unsigned int id, double value)
    {
        _dHT.insert(id, value);
        return services::Status();
    }

    services::Status find(unsigned int id, int64_t & value) const
    {
        DAAL_CHECK_EX(_iHT.find(id, value), services::ErrorHyperparameterNotFound, services::Key, id);
        return services::Status();
    }

    services::Status find(unsigned int id, double & value) const
    {
        DAAL_CHECK_EX(_dHT.find(id, value), services::ErrorHyperparameterNotFound, services::Key, id);
        return services::Status();
    }

protected:
    /** Stores integer hyperparameters of the algorithm */
    HashTable<sse2, uint32_t, int64_t> _iHT;

    /** Stores floating point hyperparameters of the algorithm */
    HashTable<sse2, uint32_t, double> _dHT;
};

} // namespace internal

namespace interface1
{
Hyperparameter::Hyperparameter(size_t intParamCount, size_t doubleParamCount)
    : _pimpl(new internal::HyperparameterImpl(intParamCount, doubleParamCount))
{}

services::Status Hyperparameter::set(unsigned int id, int64_t value)
{
    return _pimpl->set(id, value);
}

services::Status Hyperparameter::set(unsigned int id, double value)
{
    return _pimpl->set(id, value);
}

services::Status Hyperparameter::find(unsigned int id, int64_t & value) const
{
    return _pimpl->find(id, value);
}

services::Status Hyperparameter::find(unsigned int id, double & value) const
{
    return _pimpl->find(id, value);
}

} // namespace interface1
} // namespace algorithms
} // namespace daal

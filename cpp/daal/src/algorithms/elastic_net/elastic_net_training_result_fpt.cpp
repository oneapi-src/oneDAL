/* file: elastic_net_training_result_fpt.cpp */
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
//++
//  Implementation of the elastic net algorithm interface
//--
*/

#include "algorithms/elastic_net/elastic_net_training_types.h"
#include "src/algorithms/elastic_net/elastic_net_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace elastic_net
{
namespace training
{
using namespace daal::services;
/**
 * Allocates memory to store the result of elastic net model-based training
 * \param[in] input Pointer to an object containing the input data
 * \param[in] parameter %Parameter of elastic net model-based training
 * \param[in] method Computation method for the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input * input, const Parameter * parameter, const int method)
{
    const Input * const in = static_cast<const Input *>(input);

    Status s;
    const algorithmFPType dummy = 1.0;
    elastic_net::internal::ModelImpl * mImpl =
        new elastic_net::internal::ModelImpl(in->getNumberOfFeatures(), in->getNumberOfDependentVariables(), *parameter, dummy, s);
    DAAL_CHECK_MALLOC(mImpl)
    set(model, elastic_net::ModelPtr(mImpl));

    if (parameter->optResultToCompute & computeGramMatrix)
        set(gramMatrixId, data_management::HomogenNumericTable<algorithmFPType>::create(in->getNumberOfFeatures(), in->getNumberOfFeatures(),
                                                                                        data_management::NumericTableIface::doAllocate, &s));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input * input, const Parameter * parameter,
                                                                    const int method);

} // namespace training
} // namespace elastic_net
} // namespace algorithms
} // namespace daal

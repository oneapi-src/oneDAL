/* file: kdtree_knn_classification_model_impl.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
//++
//  Implementation of the class defining the K-Nearest Neighbors (kNN) model
//--
*/

#include "kdtree_knn_classification_model_impl.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kdtree_knn_classification
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_K_NEAREST_NEIGHBOR_MODEL_ID);

Model::Model(size_t nFeatures) : daal::algorithms::classifier::Model(), _impl(new ModelImpl(nFeatures)) {}

Model::~Model()
{
    delete _impl;
}

Model::Model(size_t nFeatures, services::Status &st) : _impl(new ModelImpl(nFeatures))
{
   if(!_impl) { st.add(services::ErrorMemoryAllocationFailed); }
}

services::SharedPtr<Model> Model::create(size_t nFeatures, services::Status *stat)
{
    DAAL_DEFAULT_CREATE_IMPL_EX(Model, nFeatures);
}

services::Status Model::serializeImpl(data_management::InputDataArchive  * arch)
{
    daal::algorithms::classifier::Model::serialImpl<data_management::InputDataArchive, false>(arch);
    _impl->serialImpl<data_management::InputDataArchive, false>(arch);

    return services::Status();
}

services::Status Model::deserializeImpl(const data_management::OutputDataArchive * arch)
{
    daal::algorithms::classifier::Model::serialImpl<const data_management::OutputDataArchive, true>(arch);
    _impl->serialImpl<const data_management::OutputDataArchive, true>(arch);

    return services::Status();
}

size_t Model::getNumberOfFeatures() const
{
    return _impl->getNumberOfFeatures();
}

services::Status Parameter::check() const
{
    // Inherited.
    services::Status s = daal::algorithms::classifier::Parameter::check();

    DAAL_CHECK_EX(k >= 1, services::ErrorIncorrectParameter, services::ParameterName, kStr());
    return s;
}

} // namespace interface1
} // namespace kdtree_knn_classification
} // namespace algorithms
} // namespace daal

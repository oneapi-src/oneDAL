/* file: bf_knn_classification_model_ucapi_impl.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
//  Implementation of the class defining the K-Nearest Neighbors (kNN) model
//--
*/

#ifndef __BF_KNN_CLASSIFICATION_MODEL_UCAPI_IMPL_H__
#define __BF_KNN_CLASSIFICATION_MODEL_UCAPI_IMPL_H__

#include "algorithms/k_nearest_neighbors/bf_knn_classification_model.h"
#include "data_management/data/numeric_table_sycl_homogen.h"

#include "daal_defines.h"

using namespace daal::services;
using namespace daal::oneapi::internal;

namespace daal
{
namespace algorithms
{
namespace bf_knn_classification
{
namespace interface1
{

using daal::data_management::SyclHomogenNumericTable;
using daal::data_management::NumericTable;

class Model::ModelImpl
{
public:
    /**
     * Empty constructor for deserialization
     */
    ModelImpl(size_t nFeatures = 0) : _nFeatures(nFeatures) {}


    /**
     * Returns training data
     * \return Training data
     */
    data_management::NumericTableConstPtr getData() const { return _data; }

    /**
     * Returns training data
     * \return Training data
     */
    data_management::NumericTablePtr getData() { return _data; }

    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive * arch)
    {
        arch->setSharedPtrObj(_data);
        arch->setSharedPtrObj(_labels);

        return services::Status();
    }

    /**
     * Sets a training data
     * \param[in]  value  Training data
     * \param[in]  copy   Flag indicating necessary of data deep copying to avoid direct usage and modification of input data.
     */
    template <typename algorithmFPType>
    DAAL_EXPORT DAAL_FORCEINLINE services::Status setData(const data_management::NumericTablePtr & value, bool copy)
    {
        if (!copy)
        {
            _data = value;
        }
        else
        {
            services::Status status;
            _data = SyclHomogenNumericTable<algorithmFPType>::create(value->getNumberOfColumns(), value->getNumberOfRows(),
                                                                     NumericTable::doAllocate, &status);
            DAAL_CHECK_STATUS_VAR(status);
            status |= _data->allocateDataMemory();
            data_management::BlockDescriptor<algorithmFPType> destBD, srcBD;
            status |= _data->getBlockOfRows(0, _data->getNumberOfRows(), data_management::writeOnly, destBD);
            status |= value->getBlockOfRows(0, value->getNumberOfRows(), data_management::readOnly, srcBD);
            DAAL_CHECK_STATUS_VAR(status);
            auto source = srcBD.getBuffer();
            auto destination = destBD.getBuffer();
            auto& context = Environment::getInstance()->getDefaultExecutionContext();
            context.copy(destination, 0, source, 0, source.size(), &status);
            DAAL_CHECK_STATUS_VAR(status);
            status |= _data->releaseBlockOfRows(destBD);
            status |= value->releaseBlockOfRows(srcBD);
            DAAL_CHECK_STATUS_VAR(status);
        }
        return services::Status();
    }

    /**
     * Returns training labels
     * \return Training labels
     */
    data_management::NumericTableConstPtr getLabels() const { return _labels; }

    /**
     * Returns training labels
     * \return Training labels
     */
    data_management::NumericTablePtr getLabels() { return _labels; }

    /**
     * Sets a training data
     * \param[in]  value  Training labels
     * \param[in]  copy   Flag indicating necessary of data deep copying to avoid direct usage and modification of input labels.
     */
    template <typename algorithmFPType>
    DAAL_EXPORT DAAL_FORCEINLINE services::Status setLabels(const data_management::NumericTablePtr & value, bool copy)
    {
        if (!copy)
        {
            _labels = value;
        }
        else
        {
            services::Status status;
            _labels = SyclHomogenNumericTable<algorithmFPType>::create(value->getNumberOfColumns(), value->getNumberOfRows(),
                                                                       NumericTable::doAllocate, &status);
            DAAL_CHECK_STATUS_VAR(status);
            data_management::BlockDescriptor<algorithmFPType> destBD, srcBD;
            _labels->getBlockOfRows(0, _labels->getNumberOfRows(), data_management::writeOnly, destBD);
            value->getBlockOfRows(0, value->getNumberOfRows(), data_management::readOnly, srcBD);
            DAAL_CHECK_STATUS_VAR(status);
            auto source = srcBD.getBuffer();
            auto destination = destBD.getBuffer();
            auto& context = Environment::getInstance()->getDefaultExecutionContext();
            context.copy(destination, 0, source, 0, source.size(), &status);
            DAAL_CHECK_STATUS_VAR(status);
            _labels->releaseBlockOfRows(destBD);
            value->releaseBlockOfRows(srcBD);
            DAAL_CHECK_STATUS_VAR(status);
        }
        return services::Status();

    }

    /**
     *  Retrieves the number of features in the dataset was used on the training stage
     *  \return Number of features in the dataset was used on the training stage
     */
    size_t getNumberOfFeatures() const { return _nFeatures; }
private:
    size_t _nFeatures;
    data_management::NumericTablePtr _data;
    data_management::NumericTablePtr _labels;
};


} // namespace interface1

using BFModelImplUCAPI=interface1::Model::ModelImpl;

} // namespace bf_knn_classification
} // namespace algorithms
} // namespace daal

#endif

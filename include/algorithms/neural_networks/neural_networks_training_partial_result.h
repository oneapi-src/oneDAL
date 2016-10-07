/* file: neural_networks_training_partial_result.h */
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
//  Implementation of neural network algorithm interface.
//--
*/

#ifndef __NEURAL_NETWORKS_TRAINING_PARTIAL_RESULT_H__
#define __NEURAL_NETWORKS_TRAINING_PARTIAL_RESULT_H__

#include "algorithms/algorithm.h"

#include "services/daal_defines.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/neural_networks/neural_networks_training_model.h"
#include "algorithms/neural_networks/neural_networks_training_result.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
namespace training
{
/**
 * @ingroup neural_networks_training
 * @{
 */
/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__STEP1LOCALPARTIALRESULTID"></a>
 * \brief Available identifiers of partial results of the neural network training algorithm
 * required by the first distributed step
 */
enum Step1LocalPartialResultId
{
    derivatives = 0,
    batchSize = 1
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__STEP2MASTERPARTIALRESULTID"></a>
 * \brief Available identifiers of partial results of the neural network training algorithm
 *  equired by the second distributed step
 */
enum Step2MasterPartialResultId
{
    resultFromMaster = 0
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__PARTIALRESULT"></a>
 * \brief Provides methods to access partial result obtained with the compute() method of the
 *  neural network training algorithm in the distributed processing mode
 */
class PartialResult : public daal::algorithms::PartialResult
{
public:
    DAAL_CAST_OPERATOR(PartialResult);

    PartialResult() : daal::algorithms::PartialResult(2)
    {}

    virtual ~PartialResult() {}

    /**
     * Returns partial result of the neural network model based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step1LocalPartialResultId id) const
    {
        return data_management::NumericTable::cast(Argument::get(id));
    }

    /**
     * Sets partial result of neural network model based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(Step1LocalPartialResultId id, const data_management::NumericTablePtr &value)
    {
        Argument::set(id, value);
    }

    /**
     * Registers user-allocated memory to store partial results of the neural network model based training
     * \param[in] input Pointer to an object containing %input data
     * \param[in] parameter %Parameter of the neural network training
     * \param[in] method Computation method for the algorithm
     */
    template<typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        set(batchSize, data_management::NumericTablePtr(
            new data_management::HomogenNumericTable<double>(1, 1, data_management::NumericTableIface::doAllocate)));
    }

    /**
     * Checks result of the neural network algorithm
     * \param[in] input   %Input object of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 2) { this->_errors->add(services::ErrorIncorrectNumberOfOutputNumericTables); return; }
    }

    /**
     * Returns the serialization tag of the result
     * \return         Serialization tag of the result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_TRAINING_PARTIAL_RESULT_ID; }

    /**
     *  Serializes the object
     *  \param[in]  arch  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
     *  Deserializes the object
     *  \param[in]  arch  Storage for the deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__DISTRIBUTEDPARTIALRESULT"></a>
 * \brief Provides methods to access partial result obtained with the compute() method of the
 * neural network training algorithm in the distributed processing mode
 */
class DistributedPartialResult : public daal::algorithms::PartialResult
{
public:
    DAAL_CAST_OPERATOR(DistributedPartialResult);

    DistributedPartialResult() : daal::algorithms::PartialResult(1)
    {
        set(resultFromMaster, training::ResultPtr(new Result()));
    }

    /**
     * Returns the partial result of the neural network model based training
     * \param[in] id    Identifier of the partial result
     * \return          Partial result that corresponds to the given identifier
     */
    training::ResultPtr get(Step2MasterPartialResultId id) const
    {
        return Result::cast(Argument::get(id));
    }

    /**
     * Sets the partial result of neural network model based training
     * \param[in] id      Identifier of the partial result
     * \param[in] value   Partial result
     */
    void set(Step2MasterPartialResultId id, const training::ResultPtr &value)
    {
        Argument::set(id, value);
    }

    /**
     * Registers user-allocated memory to store partial results of the neural network model based training
     * \param[in] input Pointer to an object containing %input data
     * \param[in] method Computation method for the algorithm
     * \param[in] parameter %Parameter of the neural network training
     */
    template<typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {}

    /**
     * Checks partial result of the neural network algorithm
     * \param[in] input   %Input object of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {}

    /**
     * Returns the serialization tag of the result
     * \return         Serialization tag of the result
     */
    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_NEURAL_NETWORKS_TRAINING_DISTRIBUTED_PARTIAL_RESULT_ID; }

    /**
     *  Serializes the object
     *  \param[in]  arch  Storage for the serialized object or data structure
     */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
     *  Deserializes the object
     *  \param[in]  arch  Storage for the deserialized object or data structure
     */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::OutputDataArchive, true>(arch);}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};
} // namespace interface1
typedef services::SharedPtr<interface1::PartialResult> PartialResultPtr;
typedef services::SharedPtr<interface1::DistributedPartialResult> DistributedPartialResultPtr;
typedef interface1::PartialResult PartialResult;
typedef interface1::DistributedPartialResult DistributedPartialResult;

/** @} */
}
}
}
} // namespace daal
#endif

/* file: neural_networks_training_partial_result.h */
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
    derivatives,
    batchSize,
    lastStep1LocalPartialResultId = batchSize
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__STEP2MASTERPARTIALRESULTID"></a>
 * \brief Available identifiers of partial results of the neural network training algorithm
 *  equired by the second distributed step
 */
enum Step2MasterPartialResultId
{
    resultFromMaster,
    lastStep2MasterPartialResultId = resultFromMaster
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
class DAAL_EXPORT PartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialResult);

    PartialResult();

    virtual ~PartialResult() {}

    /**
     * Returns partial result of the neural network model based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(Step1LocalPartialResultId id) const;

    /**
     * Sets partial result of neural network model based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(Step1LocalPartialResultId id, const data_management::NumericTablePtr &value);

    /**
     * Registers user-allocated memory to store partial results of the neural network model based training
     * \param[in] input Pointer to an object containing %input data
     * \param[in] parameter %Parameter of the neural network training
     * \param[in] method Computation method for the algorithm
     *
     * \return Status of computations
     */
    template<typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks result of the neural network algorithm
     * \param[in] input   %Input object of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__DISTRIBUTEDPARTIALRESULT"></a>
 * \brief Provides methods to access partial result obtained with the compute() method of the
 * neural network training algorithm in the distributed processing mode
 */
class DAAL_EXPORT DistributedPartialResult : public daal::algorithms::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(DistributedPartialResult);

    DistributedPartialResult();

    /**
     * Returns the partial result of the neural network model based training
     * \param[in] id    Identifier of the partial result
     * \return          Partial result that corresponds to the given identifier
     */
    training::ResultPtr get(Step2MasterPartialResultId id) const;

    /**
     * Sets the partial result of neural network model based training
     * \param[in] id      Identifier of the partial result
     * \param[in] value   Partial result
     */
    void set(Step2MasterPartialResultId id, const training::ResultPtr &value);

    /**
     * Registers user-allocated memory to store partial results of the neural network model based training
     * \param[in] input Pointer to an object containing %input data
     * \param[in] method Computation method for the algorithm
     * \param[in] parameter %Parameter of the neural network training
     *
     * \return Status of computations
     */
    template<typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Checks partial result of the neural network algorithm
     * \param[in] input   %Input object of algorithm
     * \param[in] par     %Parameter of algorithm
     * \param[in] method  Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return daal::algorithms::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
};

typedef services::SharedPtr<PartialResult> PartialResultPtr;
typedef services::SharedPtr<DistributedPartialResult> DistributedPartialResultPtr;
} // namespace interface1

using interface1::PartialResult;
using interface1::PartialResultPtr;
using interface1::DistributedPartialResult;
using interface1::DistributedPartialResultPtr;

/** @} */
}
}
}
} // namespace daal
#endif

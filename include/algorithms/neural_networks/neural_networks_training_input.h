/* file: neural_networks_training_input.h */
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

#ifndef __NEURAL_NETWORKS_TRAINING_INPUT_H__
#define __NEURAL_NETWORKS_TRAINING_INPUT_H__

#include "algorithms/algorithm.h"

#include "data_management/data/tensor.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_collection.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/neural_networks_training_model.h"
#include "algorithms/neural_networks/neural_networks_training_partial_result.h"

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
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__INPUTID"></a>
 * \brief Available identifiers of %input objects for the neural network model based training
 */
enum InputId
{
    data,           /*!< Training data set */
    groundTruth,    /*!< Ground-truth results for the training data set */
    lastInputId = groundTruth
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__INPUTCOLLECTIONID"></a>
 * \brief Available identifiers of %input collection objects for the neural network model based training
 */
enum InputCollectionId
{
    groundTruthCollection = lastInputId + 1,   /*!< Data collection of ground-truth results for the training data sets */
    lastInputCollectionId = groundTruthCollection
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__STEP1LOCALINPUTID"></a>
 * \brief Available identifiers of %input objects for the neural network model based training
 */
enum Step1LocalInputId
{
    inputModel  = lastInputCollectionId + 1,         /*!< Input model */
    lastStep1LocalInputId = inputModel
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__STEP2MASTERINPUTID"></a>
 * \brief Partial results from the previous steps in the distributed processing mode required by the second distributed step of the algorithm
 */
enum Step2MasterInputId
{
    partialResults,   /*!< Partial results of the neural network training algorithm computed on the first step and to be transferred  to the
                             second step in the distributed processing mode */
    lastStep2MasterInputId = partialResults
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__INPUT"></a>
 * \brief Input objects of the neural network training algorithm
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    Input(size_t nElements = lastInputCollectionId + 1);
    Input(const Input& other);

    virtual ~Input() {}

    /**
     * Returns %input object for the neural network training algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::TensorPtr get(InputId id) const;

    /**
     * Returns %input object for the neural network training algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(InputCollectionId id) const;

    /**
     * Returns %input object for the neural network training algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] key   Key to use to retrieve data
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::TensorPtr get(InputCollectionId id, size_t key) const;

    /**
     * Sets %input object for the neural network training algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the %input object
     */
    void set(InputId id, const data_management::TensorPtr &value);

    /**
     * Sets %input object for the neural network training algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the %input object
     */
    void set(InputCollectionId id, const data_management::KeyValueDataCollectionPtr &value);

    /**
     * Sets %input object for the neural network training algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] key     Key to use to retrieve data
     * \param[in] value   Pointer to the %input object
     */
    void add(InputCollectionId id, size_t key, const data_management::TensorPtr &value);

    /**
     * Checks %input object for the neural network algorithm
     * \param[in] par     Algorithm %parameter
     * \param[in] method  Computatiom method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

protected:
    services::Status checkImpl(const daal::algorithms::Parameter *par, int method) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief Input objects of the neural network training algorithm in the distributed processing mode
 */
template<ComputeStep step>
class DAAL_EXPORT DistributedInput
{};

/**
 * <a name="DAAL-CLASS-NEURAL_NETWORKS__TRAINING__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief Input objects of the neural network training algorithm in the distributed processing mode
 */
template<>
class DAAL_EXPORT DistributedInput<step1Local> : public Input
{
public:
    DistributedInput(size_t nElements = lastStep1LocalInputId + 1);
    DistributedInput(const DistributedInput& other);

    virtual ~DistributedInput() {};

    using Input::set;
    using Input::get;

    /**
     * Returns %input object for the neural network training algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    ModelPtr get(Step1LocalInputId id) const;

    /**
     * Sets %input object for the neural network training algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the %input object
     */
    void set(Step1LocalInputId id, const ModelPtr &value);

    /**
     * Checks %input object for the neural network algorithm
     * \param[in] par     Algorithm %parameter
     * \param[in] method  Computatiom method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-NEURAL_NETWORKS__TRAINING__DISTRIBUTEDINPUT_STEP2MASTER"></a>
 * \brief Input objects of the neural network training algorithm
 */
template<>
class DAAL_EXPORT DistributedInput<step2Master> : public daal::algorithms::Input
{
public:
    DistributedInput();
    DistributedInput(const DistributedInput& other);

    virtual ~DistributedInput() {};

    /**
     * Returns %input object for the neural network training algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::KeyValueDataCollectionPtr get(Step2MasterInputId id) const;

    /**
     * Sets %input object for the neural network training algorithm
     * \param[in] id      Identifier of the %input object
     * \param[in] value   Pointer to the %input object
     */
    void set(Step2MasterInputId id, const data_management::KeyValueDataCollectionPtr &value);

    /**
     * Adds input object to KeyValueDataCollection of the neural network distributed training algorithm
     * \param[in] id    Identifier of input object
     * \param[in] key   Key to use to retrieve data
     * \param[in] value Pointer to the input object value
     */
    void add(Step2MasterInputId id, size_t key, const PartialResultPtr &value);

    /**
     * Checks %input object for the neural network algorithm
     * \param[in] par     Algorithm %parameter
     * \param[in] method  Computatiom method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::Input;
using interface1::DistributedInput;

/** @} */
}
}
}
} // namespace daal
#endif

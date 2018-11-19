/* file: neural_networks_training_result.h */
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

#ifndef __NEURAL_NETWORKS_TRAINING_RESULT_H__
#define __NEURAL_NETWORKS_TRAINING_RESULT_H__

#include "algorithms/algorithm.h"

#include "data_management/data/data_serialize.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/neural_networks_training_model.h"

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
 * <a name="DAAL-ENUM-ALGORITHMS__NEURAL_NETWORKS__TRAINING__RESULTID"></a>
 * \brief Available identifiers of result of the neural network model based training
 */
enum ResultId
{
    model = 0,    /*!< Neural network model */
    lastResultId = model
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__TRAINING__RESULT"></a>
 * \brief Provides methods to access result obtained with the compute() method of the neural network training algorithm
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);

    Result();

    /**
     * Returns the result of the neural network model based training
     * \param[in] id    Identifier of the result
     * \return          Result that corresponds to the given identifier
     */
    training::ModelPtr get(ResultId id) const;

    /**
     * Sets the result of neural network model based training
     * \param[in] id      Identifier of the result
     * \param[in] value   Result
     */
    void set(ResultId id, const training::ModelPtr &value);

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
        return daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<Result> ResultPtr;
} // namespace interface1
using interface1::Result;
using interface1::ResultPtr;

}
}
/** @} */
}
} // namespace daal
#endif

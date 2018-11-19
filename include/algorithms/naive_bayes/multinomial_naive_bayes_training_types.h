/* file: multinomial_naive_bayes_training_types.h */
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
//  Naive Bayes classifier parameter structure used in the training stage
//--
*/

#ifndef __NAIVE_BAYES_TRAINING_TYPES_H__
#define __NAIVE_BAYES_TRAINING_TYPES_H__

#include "algorithms/naive_bayes/multinomial_naive_bayes_model.h"
#include "data_management/data/data_collection.h"
#include "algorithms/classifier/classifier_training_types.h"

namespace daal
{
namespace algorithms
{
namespace multinomial_naive_bayes
{
/**
 * @defgroup multinomial_naive_bayes_training Training
 * \copydoc daal::algorithms::multinomial_naive_bayes::training
 * @ingroup multinomial_naive_bayes
 * @{
 */
/**
* \brief Contains classes for training the naive Bayes model
*/
namespace training
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__METHOD"></a>
 * Available methods for computing the results of the naive Bayes algorithm
 */
enum Method
{
    defaultDense = 0, /*!< Default Training method for the multinomial naive Bayes */
    fastCSR      = 1  /*!< Training method for the multinomial naive Bayes with sparse data in CSR format */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__STEP2MASTERINPUTID"></a>
 * Available identifiers of the step 2 input
 */
enum Step2MasterInputId
{
    partialModels,                     /*!< Collection of partial models trained on local nodes */
    lastStep2MasterInputId = partialModels
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method of the
 *        naive Bayes training algorithm
 *        in the online or distributed processing
 */
class DAAL_EXPORT PartialResult : public classifier::training::PartialResult
{
public:
    DECLARE_SERIALIZABLE_CAST(PartialResult);

    PartialResult();
    virtual ~PartialResult() {}

    /**
     * Returns the partial model trained with the classification algorithm
     * \param[in] id    Identifier of the partial model, \ref classifier::training::PartialResultId
     * \return          Model trained with the classification algorithm
     */
    multinomial_naive_bayes::PartialModelPtr get(classifier::training::PartialResultId id) const;

    /**
     * Allocates memory for storing partial results of the naive Bayes training algorithm
     * \param[in] input        Pointer to input object
     * \param[in] parameter    Pointer to parameter
     * \param[in] method       Computation method
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Initializes memory for storing partial results of the naive Bayes training algorithm
     * \param[in] input        Pointer to input object
     * \param[in] parameter    Pointer to parameter
     * \param[in] method       Computation method
     *
     * \return Status of initialization
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status initialize(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
    * Returns number of columns in the naive Bayes partial result
    * \return Number of columns in the partial result
    */
    size_t getNumberOfFeatures() const;

    /**
     * Checks partial result of the naive Bayes training algorithm
     * \param[in] input      Algorithm %input object
     * \param[in] parameter  Algorithm %parameter
     * \param[in] method     Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    /**
    * Checks partial result of the naive Bayes training algorithm
    * \param[in] parameter  Algorithm %parameter
    * \param[in] method     Computation method
    *
     * \return Status of computations
    */
    services::Status check(const daal::algorithms::Parameter *parameter, int method)  const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return classifier::training::PartialResult::serialImpl<Archive, onDeserialize>(arch);
    }
    /** \private */
    services::Status checkImpl(size_t nFeatures,const daal::algorithms::Parameter* parameter) const;
};
typedef services::SharedPtr<PartialResult> PartialResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the
 *        naive Bayes training algorithm
 *        in the batch processing mode or with the finalizeCompute() method
 *       in the distributed or online processing mode
 */
class DAAL_EXPORT Result : public classifier::training::Result
{
public:
    DECLARE_SERIALIZABLE_CAST(Result);
    Result();
    virtual ~Result() {}

    /**
     * Returns the model trained with the naive Bayes training algorithm
     * \param[in] id    Identifier of the result, \ref classifier::training::ResultId
     * \return          Model trained with the classification algorithm
     */
    multinomial_naive_bayes::ModelPtr get(classifier::training::ResultId id) const;

    /**
     * Allocates memory for storing final result computed with naive Bayes training algorithm
     * \param[in] input      Pointer to input object
     * \param[in] parameter  Pointer to parameter
     * \param[in] method     Computation method
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
    * Allocates memory for storing final result computed with naive Bayes training algorithm
    * \param[in] partialResult      Pointer to partial result structure
    * \param[in] parameter          Pointer to parameter structure
    * \param[in] method             Computation method
    *
     * \return Status of computations
    */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, const int method);

    /**
    * Checks the correctness of Result object
    * \param[in] partialResult Pointer to the partial results structure
    * \param[in] parameter     Parameter of the algorithm
    * \param[in] method        Computation method
    *
     * \return Status of computations
    */
    services::Status check(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Checks the final result of the naive Bayes training algorithm
     * \param[in] input      %Input of algorithm
     * \param[in] parameter  %Parameter of algorithm
     * \param[in] method     Computation method
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        return classifier::training::Result::serialImpl<Archive, onDeserialize>(arch);
    }
    /** \private */
    services::Status checkImpl(size_t nFeatures,const daal::algorithms::Parameter* parameter) const;
};
typedef services::SharedPtr<Result> ResultPtr;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MULTINOMIAL_NAIVE_BAYES__TRAINING__DISTRIBUTEDINPUT"></a>
 * \brief Input objects of the naive Bayes training algorithm in the distributed processing mode
 */
class DAAL_EXPORT DistributedInput : public classifier::training::InputIface
{
public:
    DistributedInput();
    DistributedInput(const DistributedInput& other) : classifier::training::InputIface(other){}

    virtual ~DistributedInput() {}

    virtual size_t getNumberOfFeatures() const DAAL_C11_OVERRIDE;

    /**
     * Returns input objects of the classification algorithm in the distributed processing mode
     * \param[in] id    Identifier of the input objects
     * \return          Input object that corresponds to the given identifier
     */
    data_management::DataCollectionPtr get(Step2MasterInputId id) const;

    /**
     * Adds input object on the master node in the training stage of the classification algorithm
     * \param[in] id            Identifier of the input object
     * \param[in] partialResult Pointer to the object
     */
    void add(const Step2MasterInputId &id, const PartialResultPtr &partialResult);

    /**
     * Sets input object in the training stage of the classification algorithm
     * \param[in] id   Identifier of the object
     * \param[in] value Pointer to the object
     */
    void set(Step2MasterInputId id, const data_management::DataCollectionPtr &value);

    /**
     * Checks input parameters in the training stage of the classification algorithm
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Algorithm method
     */
    services::Status check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

} // namespace interface1
using interface1::DistributedInput;
using interface1::PartialResult;
using interface1::PartialResultPtr;
using interface1::Result;
using interface1::ResultPtr;

} // namespace training
/** @} */
} // namespace multinomial_naive_bayes
} // namespace algorithms
} // namespace daal
#endif

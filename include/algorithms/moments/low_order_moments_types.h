/* file: low_order_moments_types.h */
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
//  Definition of LowOrderMoments common types.
//--
*/

#ifndef __LOW_ORDER_MOMENTS_TYPES_H__
#define __LOW_ORDER_MOMENTS_TYPES_H__

#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
* @defgroup low_order_moments Moments of Low Order
* \copydoc daal::algorithms::low_order_moments
* @ingroup analysis
* @{
*/
/**
 * \brief Contains classes for computing the results of the low order %moments algorithm
 */
namespace low_order_moments
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__LOW_ORDER_MOMENTS__METHOD"></a>
 * Available computation methods for the low order %moments algorithm
 */
enum Method
{
    defaultDense    = 0,        /*!< Default: performance-oriented method. Works with all types of numeric tables */
    singlePassDense = 1,        /*!< Single-pass: implementation of the single-pass algorithm proposed by D.H.D. West.
                                     Supports all types of numeric tables */
    sumDense        = 2,        /*!< Precomputed sum: implementation of %moments computation algorithm in the case of a precomputed sum.
                                     Supports all types of numeric tables */
    fastCSR         = 3,        /*!< Fast: performance-oriented method. Works with Compressed Sparse Rows(CSR) numeric tables */
    singlePassCSR   = 4,        /*!< Single-pass: implementation of the single-pass algorithm proposed by D.H.D. West.
                                     Supports CSR numeric tables */
    sumCSR          = 5         /*!< Precomputed sum: implementation of the algorithm in the case of a precomputed sum.
                                     Supports CSR numeric tables */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LOW_ORDER_MOMENTS__INPUTID"></a>
 * Available identifiers of input objects for the low order %moments algorithm
 */
enum InputId
{
    data = 0                    /*!< %Input data table */
};

const size_t nResults = 10;

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LOW_ORDER_MOMENTS__RESULTID"></a>
 * Available identifiers of the results of the low order %moments algorithm
 */
enum ResultId
{
    minimum              = 0,   /*!< Minimum */
    maximum              = 1,   /*!< Maximum */
    sum                  = 2,   /*!< Sum */
    sumSquares           = 3,   /*!< Sum of squares */
    sumSquaresCentered   = 4,   /*!< Sum of squared difference from the means */
    mean                 = 5,   /*!< Mean */
    secondOrderRawMoment = 6,   /*!< Second raw order moment */
    variance             = 7,   /*!< Variance */
    standardDeviation    = 8,   /*!< Standard deviation */
    variation            = 9    /*!< Variation */
};

const size_t nPartialResults = 6;

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LOW_ORDER_MOMENTS__PARTIALRESULTID"></a>
 * Available identifiers of partial results of the low order %moments algorithm
 */
enum PartialResultId
{
    nObservations               = 0,    /*!< Number of observations processed so far */
    partialMinimum              = 1,    /*!< Partial minimum */
    partialMaximum              = 2,    /*!< Partial maximum */
    partialSum                  = 3,    /*!< Partial sum */
    partialSumSquares           = 4,    /*!< Partial sum of squares */
    partialSumSquaresCentered   = 5     /*!< Partial sum of squared difference from the means */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__LOW_ORDER_MOMENTS__MASTERINPUTID"></a>
 * \brief Available identifiers of input objects for the low order moments algorithm on the master node
 */
enum MasterInputId
{
    partialResults = 0 /*!< Collection of partial results computed on local nodes */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__INPUTIFACE"></a>
 * \brief Abstract class that specifies interface of the input objects for the low order %moments algorithm
 */
class InputIface : public daal::algorithms::Input
{
public:
    InputIface(size_t nElements) : daal::algorithms::Input(nElements) {}
    virtual size_t getNumberOfColumns() const = 0;
    virtual ~InputIface() {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__INPUT"></a>
 * \brief %Input objects for the low order %moments algorithm
 */
class DAAL_EXPORT Input : public InputIface
{
public:
    Input();

    virtual ~Input() {};

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNumberOfColumns() const DAAL_C11_OVERRIDE;

    /**
     * Returns the input object for the low order %moments algorithm
     * \param[in] id    Identifier of the %input object
     * \return          %Input object that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets input object for the low order %moments algorithm
     * \param[in] id    Identifier of the %input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__PARTIALRESULT"></a>
 * \brief Provides methods to access partial results obtained with the compute() method
 *        of the low order %moments algorithm
 *        in the online or distributed processing mode
 */
class DAAL_EXPORT PartialResult : public daal::algorithms::PartialResult
{
public:

    DAAL_CAST_OPERATOR(PartialResult);

    PartialResult();

    virtual ~PartialResult()
    {}

    /**
     * Allocates memory to store partial results of the low order %moments algorithm
     * \param[in] input     Pointer to the structure with input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Gets the number of columns in the partial result of the low order %moments algorithm
     * \return Number of columns in the partial result
     */
    size_t getNumberOfColumns() const;

    /**
     * Returns the partial result of the low order %moments algorithm
     * \param[in] id   Identifier of the partial result, \ref PartialResultId
     * \return Partial result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(PartialResultId id) const;

    /**
     * Sets the partial result of the low order %moments algorithm
     * \param[in] id    Identifier of the partial result
     * \param[in] ptr   Pointer to the partial result
     */
    void set(PartialResultId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks correctness of the partial result
     * \param[in] parameter %Parameter of the algorithm
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    /**
     * Checks  the correctness of partial result
     * \param[in] input     Pointer to the structure with input objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_MOMENTS_PARTIAL_RESULT_ID; }

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

    void checkImpl(size_t nFeatures) const;
};

typedef services::SharedPtr<PartialResult> PartialResultPtr;

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__LOW_ORDER_MOMENTS__PARTIALRESULTSINITIFACE"></a>
 * \brief Abstract interface class for partial results initialization
 */
struct DAAL_EXPORT PartialResultsInitIface : public Base
{
    /**
     * Initializes partial results of the low order %moments algorithm
     * \param[in]       input     %Input objects for the algorithm
     * \param[in,out]   pres      Partial results of the algorithm
     */
    virtual void operator()(const Input &input, services::SharedPtr<PartialResult> &pres) = 0;
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__LOW_ORDER_MOMENTS__DEFAULTPARTIALRESULTINIT"></a>
 * \brief Class that specifies the default method for initialization of partial results
 */
struct DAAL_EXPORT DefaultPartialResultInit : public PartialResultsInitIface
{
    /**
     * Initializes partial results
     * \param[in]       input     %Input objects for the algorithm
     * \param[in,out]   pres      Partial results of the algorithm
     */
    virtual void operator()(const Input &input, services::SharedPtr<PartialResult> &pres);
};

/**
 * <a name="DAAL-STRUCT-ALGORITHMS__LOW_ORDER_MOMENTS__PARAMETER"></a>
 * \brief Low order %moments algorithm parameters
 */
struct DAAL_EXPORT Parameter : public daal::algorithms::Parameter
{
    /** Constructs default low order %moments parameters */
    Parameter();
    services::SharedPtr<PartialResultsInitIface> initializationProcedure;     /**< Functor for partial results initialization */

    void check() const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the low order %moments algorithm in the batch processing mode
 *        ; or finalizeCompute() method of algorithm in the online or distributed processing mode
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    Result();

    virtual ~Result() {};

    /**
     * Allocates memory for storing final results of the low order %moments algorithm
     * \param[in] input     Pointer to the structure with result objects
     * \param[in] parameter Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

    /**
     * Allocates memory for storing final results of the low order %moments algorithm
     * \param[in] partialResult     Pointer to the structure with partial result objects
     * \param[in] parameter         Pointer to the structure of algorithm parameters
     * \param[in] method            Computation method
     */
    template <typename algorithmFPType>
    DAAL_EXPORT void allocate(const daal::algorithms::PartialResult *partialResult, daal::algorithms::Parameter *parameter, const int method);

    /**
     * Returns final result of the low order %moments algorithm
     * \param[in] id   identifier of the result, \ref ResultId
     * \return         Final result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets final result of the low order %moments algorithm
     * \param[in] id    Identifier of the final result
     * \param[in] value Pointer to the final result
     */
    void set(ResultId id, const data_management::NumericTablePtr &value);

    /**
     * Checks the correctness of result
     * \param[in] partialResult Pointer to the partial results
     * \param[in] par           %Parameter of the algorithm
     * \param[in] method        Computation method
     */
    void check(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    /**
     * Checks the correctness of result
     * \param[in] input     Pointer to the structure with input objects
     * \param[in] par       Pointer to the structure of algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;

    int getSerializationTag() DAAL_C11_OVERRIDE  { return SERIALIZATION_MOMENTS_RESULT_ID; }

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
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);
    }

    void checkImpl(size_t nFeatures) const;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTEDINPUT"></a>
 * \brief Parameters of the low order moments algorithm in the distributed processing mode.
 *
 * \tparam step             Step of distributed processing, \ref ComputeStep
 */
template<ComputeStep step>
class DistributedInput {};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTEDINPUT_STEP1LOCAL"></a>
 * \brief Local-node input objects for the low order moments algorithm in the distributed processing mode.
 */
template<>
class DAAL_EXPORT DistributedInput<step1Local> : public Input
{
public:
    DistributedInput() : Input()
    {}

    virtual ~DistributedInput()
    {}
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOW_ORDER_MOMENTS__DISTRIBUTEDINPUT_STEP2MASTER"></a>
 * \brief Input objects for the low order moments algorithm in the distributed processing mode on master node.

 */
template<>
class DAAL_EXPORT DistributedInput<step2Master> : public InputIface
{
public:
    DistributedInput();

    virtual ~DistributedInput() {}

    /**
     * Returns the number of columns in the input data set
     * \return Number of columns in the input data set
     */
    size_t getNumberOfColumns() const DAAL_C11_OVERRIDE;

    /**
     * Adds partial result to the collection of input objects for the low order moments algorithm in the distributed processing mode.
     * \param[in] id            Identifier of the input object
     * \param[in] partialResult Partial result obtained in the first step of the distributed algorithm
     */
    void add(MasterInputId id, const services::SharedPtr<PartialResult> &partialResult);

    /**
     * Sets input object for the low order moments algorithm in the distributed processing mode.
     * \param[in] id  Identifier of the input object
     * \param[in] ptr Pointer to the input object
     */
    void set(MasterInputId id, const data_management::DataCollectionPtr &ptr);

    /**
     * Returns the collection of input objects
     * \param[in] id   Identifier of the input object, \ref MasterInputId
     * \return Collection of distributed input objects
     */
    data_management::DataCollectionPtr get(MasterInputId id) const;

    /**
     * Checks algorithm parameters on the master node
     * \param[in] parameter Pointer to the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;
};
/** @} */
} // namespace interface1
using interface1::InputIface;
using interface1::Input;
using interface1::PartialResultPtr;
using interface1::PartialResult;
using interface1::PartialResultsInitIface;
using interface1::DefaultPartialResultInit;
using interface1::Parameter;
using interface1::Result;
using interface1::DistributedInput;

} // namespace daal::algorithms::low_order_moments
}
}
#endif

/* file: engine_types.h */
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
//  Implementation of engine.
//--
*/

#ifndef __ENGINE__TYPES__H__
#define __ENGINE__TYPES__H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup engines Engines
 * \copydoc daal::algorithms::engines
 * @ingroup analysis
 * @{
 */
/**
 * \brief Contains classes for engines
 */
namespace engines
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__ENGINES__INPUTID"></a>
 * Available identifiers of input objects for engines
 */
enum InputId
{
    tableToFill = 0    /*!< Input table to fill with random numbers */
};

/**
 * <a name="DAAL-ENUM-ALGORITHMS__ENGINES__RESULTID"></a>
 * Available identifiers of results for the engines
 */
enum ResultId
{
    randomNumbers = 0  /*!< Table to store the result */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__INPUT"></a>
 * \brief %Input objects for engines
 */
class DAAL_EXPORT Input : public daal::algorithms::Input
{
public:
    /**
     * Default constructor
     */
    Input();
    /** Copy constructor */
    Input(const Input& other);

    virtual ~Input() {}

    /**
     * Returns input table of the engine
     * \param[in] id    Identifier of the input table
     * \return          %Input table that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(InputId id) const;

    /**
     * Sets input for the engine
     * \param[in] id    Identifier of the input object
     * \param[in] ptr   Pointer to the object
     */
    void set(InputId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks an input object for the engine
     * \param[in] par     %Parameter of the algorithm
     * \param[in] method  Computation method of the algorithm
     *
     * \return Status of computations
     */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ENGINES__RESULT"></a>
 * \brief Provides methods to access the result obtained with the compute() method of the engine
 */
class DAAL_EXPORT Result : public daal::algorithms::Result
{
public:
    /** \brief Constructor */
    Result();

    virtual ~Result() {}

    /**
     * Allocates memory to store the results of the engine
     * \param[in] input  Pointer to the input structure
     * \param[in] par    Pointer to the parameter structure
     * \param[in] method Computation method of the algorithm
     *
     * \return Status of computations
     */
    template <typename algorithmFPType>
    DAAL_EXPORT services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, const int method);

    /**
     * Returns result of the engine
     * \param[in] id   Identifier of the result
     * \return         Result that corresponds to the given identifier
     */
    data_management::NumericTablePtr get(ResultId id) const;

    /**
     * Sets the result of the engine
     * \param[in] id    Identifier of the result
     * \param[in] ptr   Pointer to the result
     */
    void set(ResultId id, const data_management::NumericTablePtr &ptr);

    /**
     * Checks the result object for the engine
     * \param[in] input         %Input of the algorithm
     * \param[in] parameter     %Parameter of algorithm
     * \param[in] method        Computation method of the algorithm
     *
     * \return Status of computations
     */
    virtual services::Status check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, int method) const DAAL_C11_OVERRIDE;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    services::Status serialImpl(Archive *arch)
    {
        daal::algorithms::Result::serialImpl<Archive, onDeserialize>(arch);

        return services::Status();
    }

    services::Status serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<data_management::InputDataArchive, false>(arch);

        return services::Status();
    }

    services::Status deserializeImpl(const data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    {
        serialImpl<const data_management::OutputDataArchive, true>(arch);

        return services::Status();
    }
};
typedef services::SharedPtr<Result> ResultPtr;
} // interface1
using interface1::Input;
using interface1::Result;
using interface1::ResultPtr;
} // namespace engines
/** @} */
} // namespace algorithm
} // namespace daal
#endif

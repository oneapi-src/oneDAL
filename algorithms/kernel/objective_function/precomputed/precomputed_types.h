/* file: precomputed_types.h */
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
//  Implementation of sum of loss objective function interface.
//--
*/

#ifndef __PRECOMPUTED_TYPES_H__
#define __PRECOMPUTED_TYPES_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "sum_of_functions_types.h"
#include "numeric_table.h"



namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for computing the sum of loss objective function
 */
namespace optimization_solver
{
namespace internal
{
namespace precomputed
{
enum Method
{
    defaultDense = 0
};
namespace interface1
{

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__precomputed__PARAMETER"></a>
 * \brief %Parameter for sum of loss objective function
 *
 * \snippet optimization_solver/objective_function/precomputed_types.h Parameter source code
 */
/* [Parameter source code] */
struct DAAL_EXPORT Parameter : public sum_of_functions::Parameter
{
    /**
     * Constructs the parameter of sum of loss objective function
     * \param[in] numberOfTerms    The number of terms in the function
     * \param[in] batchIndices     Numeric table of size 1 x m where m is batch size that represent
                                   a batch of indices used to compute the function results, e.g.,
                                   value of the sum of the functions. If no indices are provided,
                                   all terms will be used in the computations.
     * \param[in] resultsToCompute 64 bit integer flag that indicates the results to compute
     */
    Parameter(size_t numberOfTerms,
              data_management::NumericTablePtr batchIndices = data_management::NumericTablePtr(),
              const DAAL_UINT64 resultsToCompute = objective_function::gradient);

    /**
     * Constructs an Parameter by copying input objects and parameters of another Parameter
     * \param[in] other An object to be used as the source to initialize object
     */
    Parameter(const Parameter &other);

    /**
     * Checks the correctness of the parameter
     */
    virtual void check() const
    {
        sum_of_functions::Parameter::check();
    }

    virtual ~Parameter() {}
};
/* [Parameter source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__precomputed__INPUT"></a>
 * \brief %Input objects for the sum of loss objective function
 */
class Input : public sum_of_functions::Input
{
public:
    /** Default constructor */
    Input() : sum_of_functions::Input(1)
    {}

    /** Destructor */
    virtual ~Input() {}

    /**
     * Checks the correctness of the input
     * \param[in] par       Pointer to the structure of the algorithm parameters
     * \param[in] method    Computation method
     */
    void check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        if(Argument::size() != 1)
        { this->_errors->add(services::ErrorIncorrectNumberOfInputNumericTables); return; }
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__OBJECTIVE_FUNCTION__RESULT"></a>
 * \brief Provides methods to access final results obtained with the compute() method of the Objective function in the batch processing mode
 */
class Result: public objective_function::Result
{
public:
    /** Default constructor */
    Result() : objective_function::Result()
    {}

    /** Destructor */
    virtual ~Result() {};

    /**
     * Allocates memory for storing results of the Objective function
     * \param[in] input     Pointer to the input structure
     * \param[in] parameter Pointer to the parameter structure
     * \param[in] method    Computation method
     */
    template <typename algorithmFPType>
    void allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
    {
        using namespace services;
        using namespace data_management;

        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(parameter);
        DAAL_CHECK(algParameter != 0, ErrorNullParameterNotSupported);

        size_t nRows = algInput->get(sum_of_functions::argument)->getNumberOfRows();

        if(algParameter->resultsToCompute & objective_function::gradient)
        {
            NumericTablePtr nt = NumericTablePtr(new HomogenNumericTable<algorithmFPType>(1, nRows, NumericTable::doAllocate));
            Argument::set(objective_function::gradientIdx, nt);
        }
        if(algParameter->resultsToCompute & objective_function::value)
        {
            NumericTablePtr nt = NumericTablePtr(new HomogenNumericTable<algorithmFPType>(1, 1, NumericTable::doAllocate));
            Argument::set(objective_function::valueIdx, nt);
        }
    }

    /**
    * Checks the result of the Objective function
    * \param[in] input   %Input of the algorithm
    * \param[in] par     %Parameter of algorithm
    * \param[in] method  Computation method
    */
    virtual void check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE
    {
        using namespace services;
        using namespace data_management;

        if(Argument::size() != 3) { this->_errors->add(ErrorIncorrectNumberOfArguments); return; }

        const Input *algInput = static_cast<const Input *>(input);
        const Parameter *algParameter = static_cast<const Parameter *>(par);
        DAAL_CHECK(algParameter != 0, ErrorNullParameterNotSupported);

        size_t nRows = algInput->get(sum_of_functions::argument)->getNumberOfRows();

        if(algParameter->resultsToCompute & objective_function::value)
        {
            if(!checkNumericTable(get(objective_function::valueIdx).get(), this->_errors.get(), valueStr(), 0, 0, 1, 1)) {return;}
        }
        if(algParameter->resultsToCompute & objective_function::gradient)
        {
            if(!checkNumericTable(get(objective_function::gradientIdx).get(), this->_errors.get(), gradientStr(), 0, 0, 1, nRows)) {return;}
        }
        DAAL_CHECK_EX((algParameter->resultsToCompute & objective_function::hessian) == 0, services::ErrorNullOutputNumericTable, \
                      services::ArgumentName, "Hessian for sum of loss is not supported");
    }

    /**
     * Returns a serialization tag, a unique identifier of this class used in serialization
     * \return Serialization tag
     */
    int getSerializationTag() DAAL_C11_OVERRIDE { return SERIALIZATION_OBJECTIVE_FUNCTION_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for a serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    {serialImpl<data_management::InputDataArchive, false>(arch);}

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for a deserialized object or data structure
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
};

} // namespace interface1
using interface1::Parameter;
using interface1::Input;
using interface1::Result;

} // namespace precomputed
} // namespace internal
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif

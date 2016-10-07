/* file: algorithm_types.h */
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
//  Implementation of base classes defining algorithm interface.
//--
*/

#ifndef __ALGORITHM_TYPES_H__
#define __ALGORITHM_TYPES_H__

#include "services/daal_defines.h"
#include "data_management/data/data_archive.h"
#include "data_management/data/data_serialize.h"
#include "data_management/data/data_collection.h"
#include "services/error_handling.h"

namespace daal
{
/**
 * @defgroup base_algorithms Base Classes
 * \copydoc daal::algorithms
 * @ingroup algorithms
 * @{
 */
/**
 * \brief Contains classes that implement algorithms for data analysis(data mining), and data modeling(training and prediction).
 *        These algorithms include matrix decompositions, clustering algorithms, classification and regression algorithms,
 *        as well as association rules discovery.
 */
namespace algorithms
{

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 *  <a name="DAAL-STRUCT-ALGORITHMS__PARAMETER"></a>
 *  \brief %Base class to represent computation parameters.
 *         Algorithm-specific parameters are represented as derivative classes of the Parameter class.
 */
struct Parameter
{
    DAAL_NEW_DELETE();

    Parameter() : _errors(new services::ErrorCollection()) {}
    virtual ~Parameter() {}

    virtual void check() const {}

    /**
     * Sets the collection of errors
     * \param[in] errors    Pointer to the collection of errors
     */
    void setErrorCollection(const services::ErrorCollectionPtr& errors)
    {
        _errors = errors;
    }

    /**
     * Resturns the collection of errors
     * \return  Pointer to the collection of errors
     */
    services::ErrorCollectionPtr getErrorCollection()
    {
        return _errors;
    }

    services::ErrorCollectionPtr _errors;
};

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__ARGUMENT"></a>
 *  \brief %Base class to represent computation input and output arguments.
 */
class DAAL_EXPORT Argument
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor. Constructs empty arguments */
    Argument() : _errors(new services::ErrorCollection()), idx(0) {}

    /**
     * Constructs conputation argument that contains n elements
     * \param[in] n     Number of elements in the argument
     */
    Argument(const size_t n);

    virtual ~Argument() {};

    /**
     * Inserts element into this argument structure
     * \param[in] val   Element to insert
     * \return Updated argument structure
     */
    Argument &operator <<(const data_management::SerializationIfacePtr &val)
    {
        (*_storage) << val;
        return *this;
    }

    /**
     * Retrieves number of elements in the argument
     * \return  Number of elements in the argument
     */
    size_t size() const
    {
        return _storage->size();
    }

    /**
     * Sets the collection of errors
     * \param[in] errors    Pointer to the collection of errors
     */
    void setErrorCollection(const services::ErrorCollectionPtr& errors)
    {
        _errors = errors;
    }

protected:
    /**
     * Retrieves specified element
     * \param[in] index Index of the element
     * \return Reference to the requested element
     */
    const data_management::SerializationIfacePtr& get(size_t index) const;

    /**
     * Sets the element to the specified position in the Argument
     * \param[in] index Index of the element
     * \param[in] value Pointer to the element
     * \return Reference to the requested element
     */
    void set(size_t index, const data_management::SerializationIfacePtr &value);

    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        arch->set(idx);
        arch->setObj(_storage.get());
    }

    services::ErrorCollectionPtr _errors;

private:
    size_t idx;
    data_management::DataCollectionPtr _storage;
};

/**
*  <a name="DAAL-CLASS-ALGORITHMS__SERIALIZABLE_ARGUMENT"></a>
*  \brief %Base class to represent argument with serialization methods
*/
class SerializableArgument : public data_management::SerializationIface, public Argument
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor. Constructs empty argument */
    SerializableArgument(){}

    /**
    * Constructs argument with n elements
    * \param[in] n     Number of elements in the input argument
    */
    SerializableArgument(const size_t n) : Argument(n) {}

    virtual ~SerializableArgument() {};
};

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__INPUT"></a>
 *  \brief %Base class to represent computation input arguments.
 *         Algorithm-specific input arguments are represented as derivative classes of the Input class.
 */
class Input : public data_management::SerializationIface, public Argument
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor. Constructs empty input arguments */
    Input(){}

    /**
     * Constructs input arguments with n elements
     * \param[in] n     Number of elements in the input argument
     */
    Input(const size_t n) : Argument(n), data_management::SerializationIface() {}

    virtual ~Input() {};

    virtual int getSerializationTag() { return 0; }

    virtual void serializeImpl(data_management::InputDataArchive *archive) {}
    virtual void deserializeImpl(data_management::OutputDataArchive *archive) {}

    /**
     * Checks the correctness of the input arguments
     * \param[in] parameter     Pointer to the parameters of the algorithm
     * \param[in] method        Computation method
     */
    virtual void check(const Parameter *parameter, int method) const {}
};

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__PARTIALRESULT"></a>
 *  \brief %Base class to represent partial results of the computation.
 *         Algorithm-specific partial results are represented as derivative classes of the PartialResult class.
 */
class PartialResult :public data_management::SerializationIface, public Argument
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor. Constructs empty partial results */
    PartialResult() : _initFlag(false) {};

    /**
     * Constructs partial results with n elements
     * \param[in] n     Number of elements in the partial results
     */
    PartialResult(const size_t n) : Argument(n), data_management::SerializationIface(), _initFlag(false) {}

    virtual ~PartialResult() {};

    virtual int getSerializationTag() { return 0; }
    virtual void serializeImpl(data_management::InputDataArchive *archive) {}
    virtual void deserializeImpl(data_management::OutputDataArchive *archive) {}

    /**
     * Retrieves the initialization flag
     * \return Initialization flag. True, if the partial results are already initialized; false - otherwise.
     */
    bool getInitFlag() { return _initFlag; }

    /**
     * Sets the initialization flag
     * \param[in] flag  Initialization flag. True, if the partial results are already initialized; false - otherwise.
     */
    void setInitFlag(bool flag) { _initFlag = flag; }

    /**
     * Checks the correctness of the partial results structure
     * \param[in] input         Pointer to the input arguments of the algorithm
     * \param[in] parameter     Pointer to the parameters of the algorithm
     * \param[in] method        Computation method
     */
    virtual void check(const Input *input,
                       const Parameter *parameter,
                       int method) const {}

    /**
    * Checks the correctness of the partial results structure
    * \param[in] parameter     Pointer to the parameters of the algorithm
    * \param[in] method        Computation method
    */
    virtual void check(const Parameter *parameter,
                       int method) const {}

private:
    bool _initFlag;

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        Argument::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
 *  <a name="DAAL-CLASS-ALGORITHMS__RESULT"></a>
 *  \brief %Base class to represent final results of the computation.
 *         Algorithm-specific final results are represented as derivative classes of the Result class.
 */
class Result : public data_management::SerializationIface, public Argument
{
public:
    DAAL_NEW_DELETE();

    /** Default constructor. Constructs empty final results */
    Result(){}

    /**
     * Constructs final results with n elements
     * \param[in] n     Number of elements in the final results
     */
    Result(const size_t n) : Argument(n), data_management::SerializationIface() {}

    virtual ~Result() {};

    virtual int getSerializationTag() { return 0; }
    virtual void serializeImpl(data_management::InputDataArchive *archive) {}
    virtual void deserializeImpl(data_management::OutputDataArchive *archive) {}

    /**
     * Checks the correctness of the final results structure
     * \param[in] input         Pointer to the input arguments of the algorithm
     * \param[in] parameter     Pointer to the parameters of the algorithm
     * \param[in] method        Computation method
     */
    virtual void check(const Input *input,
                       const Parameter *parameter,
                       int method) const {}

    /**
    * Checks the correctness of the partial result structure
    * \param[in] partialResult Pointer to the partial result arguments of the algorithm
    * \param[in] parameter     Pointer to the parameters of the algorithm
    * \param[in] method        Computation method
    */
    virtual void check(const PartialResult *partialResult,
                       const Parameter *parameter,
                       int method) const {}

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        Argument::serialImpl<Archive, onDeserialize>(arch);
    }
};

/**
*  <a name="DAAL-CLASS-ALGORITHMS__OPTIONAL_ARGUMENT"></a>
*  \brief %Base class to represent argument with serialization methods
*/
class OptionalArgument : public SerializableArgument
{
public:
    /** Default constructor. Constructs empty argument */
    OptionalArgument(){}

    /**
    * Constructs argument with n elements
    * \param[in] n     Number of elements in the input argument
    */
    OptionalArgument(const size_t n) : SerializableArgument(n) {}

    /**
    * Retrieves specified element
    * \param[in] index Index of the element
    * \return Reference to the requested element
    */
    const data_management::SerializationIfacePtr& get(size_t index) const { return SerializableArgument::get(index); }

    /**
    * Sets the element to the specified position in the Argument
    * \param[in] index Index of the element
    * \param[in] value Pointer to the element
    * \return Reference to the requested element
    */
    void set(size_t index, const data_management::SerializationIfacePtr &value) { return SerializableArgument::set(index, value); }

    /**
    * Returns a serialization tag, a unique identifier of this class used in serialization
    * \return Serialization tag
    */
    int getSerializationTag() DAAL_C11_OVERRIDE{ return SERIALIZATION_OPTIONAL_RESULT_ID; }

    /**
    *  Serializes the object
    *  \param[in]  arch  Storage for the serialized object or data structure
    */
    void serializeImpl(data_management::InputDataArchive  *arch) DAAL_C11_OVERRIDE
    { serialImpl<data_management::InputDataArchive, false>(arch); }

    /**
    *  Deserializes the object
    *  \param[in]  arch  Storage for the deserialized object or data structure
    */
    void deserializeImpl(data_management::OutputDataArchive *arch) DAAL_C11_OVERRIDE
    { serialImpl<data_management::OutputDataArchive, true>(arch); }

protected:
    /** \private */
    template<typename Archive, bool onDeserialize>
    void serialImpl(Archive *arch)
    {
        Argument::serialImpl<Archive, onDeserialize>(arch);
    }
};
typedef services::SharedPtr<OptionalArgument> OptionalArgumentPtr;

/** @} */
} // namespace interface1
using interface1::Parameter;
using interface1::Argument;
using interface1::Input;
using interface1::PartialResult;
using interface1::Result;
using interface1::OptionalArgument;
using interface1::OptionalArgumentPtr;

}
}
#endif

/* file: helpers.h */
/*******************************************************************************
* Copyright 2014 Intel Corporation
* Copyright contributors to the oneDAL project
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

#ifndef __DATA_MANAGEMENT_FEATURES_INTERNAL_HELPERS_H__
#define __DATA_MANAGEMENT_FEATURES_INTERNAL_HELPERS_H__

#include "services/internal/utilities.h"
#include "services/internal/collection.h"

#include "data_management/features/indices.h"

namespace daal
{
namespace data_management
{
namespace features
{
namespace internal
{
template <typename T>
inline services::Status pickElementsRaw(const FeatureIndicesIfacePtr & indices, T * elements, T ** pickedElements)
{
    DAAL_ASSERT(indices);
    DAAL_ASSERT(elements);
    DAAL_ASSERT(pickedElements);

    if (indices->isPlainRange())
    {
        const size_t first = indices->getFirst();
        const size_t last  = indices->getLast();

        size_t k = 0;
        if (first <= last)
        {
            for (size_t i = first; i <= last; i++)
            {
                pickedElements[k++] = &elements[i];
            }
        }
        else
        {
            for (size_t i = first + 1; i > last; i--)
            {
                pickedElements[k++] = &elements[i - 1];
            }
        }
    }
    else if (indices->areRawFeatureIndicesAvailable())
    {
        const services::BufferView<FeatureIndex> indicesBuffer = indices->getRawFeatureIndices();
        const FeatureIndex * rawIndices                        = indicesBuffer.data();
        const size_t indicesSize                               = indicesBuffer.size();

        for (size_t i = 0; i < indicesSize; i++)
        {
            pickedElements[i] = &elements[rawIndices[i]];
        }
    }
    else
    {
        return services::throwIfPossible(services::ErrorMethodNotImplemented);
    }

    return services::Status();
}

template <typename T>
inline services::internal::CollectionPtr<T *> pickElements(const FeatureIndicesIfacePtr & indices, T * elements, services::Status * status = NULL)
{
    DAAL_ASSERT(indices);
    DAAL_ASSERT(elements);

    services::internal::CollectionPtr<T *> pickedElements = services::internal::HeapAllocatableCollection<T *>::create(indices->size(), status);
    if (!pickedElements)
    {
        return pickedElements;
    }

    services::Status pickElementsStatus = pickElementsRaw<T>(indices, elements, pickedElements->data());
    if (!pickElementsStatus.ok())
    {
        services::internal::tryAssignStatusAndThrow(status, pickElementsStatus);
        return services::internal::CollectionPtr<T *>();
    }

    return pickedElements;
}

template <typename T>
inline services::internal::CollectionPtr<T *> pickElements(const FeatureIndicesIfacePtr & indices, const services::Collection<T> & elements,
                                                           services::Status * status = NULL)
{
    return pickElements(indices, const_cast<T *>(elements.data()), status);
}

template <typename T>
inline services::internal::CollectionPtr<T *> pickElements(const FeatureIndicesIfacePtr & indices,
                                                           const services::internal::CollectionPtr<T> & elements, services::Status * status = NULL)
{
    DAAL_ASSERT(elements);
    return pickElements(indices, const_cast<T *>(elements->data()), status);
}

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__FEATURES__INTERNAL__ELEMENTSPICKER"></a>
 * \brief Class that stores collection of elements of specified type and pointers to the elements
 *        of that collection corresponding to the indices provided in ElementsPicker::pick method
 * \tparam T Type of elements in collection
 */
template <typename T>
class ElementsPicker
{
public:
    services::Status pick(const FeatureIndicesIfacePtr & indices)
    {
        DAAL_ASSERT(indices);
        DAAL_ASSERT(_elements);

        services::Status status;
        _pickedElements = pickElements(indices, _elements, &status);

        return status;
    }

    void setElements(const services::internal::CollectionPtr<T> & elements)
    {
        DAAL_ASSERT(elements);
        _elements = elements;
    }

    const services::internal::CollectionPtr<T> & getElements() const { return _elements; }

    const services::internal::CollectionPtr<T *> & getPickedElements() const { return _pickedElements; }

private:
    services::internal::CollectionPtr<T> _elements;
    services::internal::CollectionPtr<T *> _pickedElements;
};

/**
 * Convert from a given C++ type to InternalNumType
 * \return Converted numeric type
 */
template <typename T>
inline IndexNumType getIndexNumType()
{
    return DAAL_OTHER_T;
}
template <>
inline IndexNumType getIndexNumType<float>()
{
    return DAAL_FLOAT32;
}
template <>
inline IndexNumType getIndexNumType<double>()
{
    return DAAL_FLOAT64;
}
template <>
inline IndexNumType getIndexNumType<int>()
{
    return DAAL_INT32_S;
}
template <>
inline IndexNumType getIndexNumType<unsigned int>()
{
    return DAAL_INT32_U;
}
template <>
inline IndexNumType getIndexNumType<DAAL_INT64>()
{
    return DAAL_INT64_S;
}
template <>
inline IndexNumType getIndexNumType<DAAL_UINT64>()
{
    return DAAL_INT64_U;
}
template <>
inline IndexNumType getIndexNumType<char>()
{
    return DAAL_INT8_S;
}
template <>
inline IndexNumType getIndexNumType<unsigned char>()
{
    return DAAL_INT8_U;
}
template <>
inline IndexNumType getIndexNumType<short>()
{
    return DAAL_INT16_S;
}
template <>
inline IndexNumType getIndexNumType<unsigned short>()
{
    return DAAL_INT16_U;
}

template <>
inline IndexNumType getIndexNumType<long>()
{
    return (IndexNumType)(DAAL_INT32_S + (sizeof(long) / 4 - 1) * 2);
}

#if (defined(__APPLE__) || defined(__MACH__)) && !defined(__x86_64__)
template <>
inline IndexNumType getIndexNumType<unsigned long>()
{
    return (IndexNumType)(DAAL_INT32_U + (sizeof(unsigned long) / 4 - 1) * 2);
}
#endif

#if !(defined(_WIN32) || defined(_WIN64)) && (defined(__x86_64__) || defined(TARGET_ARM) || defined(TARGET_RISCV64))
template <>
inline IndexNumType getIndexNumType<size_t>()
{
    return (IndexNumType)(DAAL_INT32_U + (sizeof(size_t) / 4 - 1) * 2);
}
#endif

/**
 * \return PMMLNumType
 */
template <typename T>
inline PMMLNumType getPMMLNumType()
{
    return DAAL_GEN_UNKNOWN;
}
template <>
inline PMMLNumType getPMMLNumType<int>()
{
    return DAAL_GEN_INTEGER;
}
template <>
inline PMMLNumType getPMMLNumType<double>()
{
    return DAAL_GEN_DOUBLE;
}
template <>
inline PMMLNumType getPMMLNumType<float>()
{
    return DAAL_GEN_FLOAT;
}
template <>
inline PMMLNumType getPMMLNumType<bool>()
{
    return DAAL_GEN_BOOLEAN;
}
template <>
inline PMMLNumType getPMMLNumType<char *>()
{
    return DAAL_GEN_STRING;
}
template <>
inline PMMLNumType getPMMLNumType<std::string>()
{
    return DAAL_GEN_STRING;
}

} // namespace internal
} // namespace features
} // namespace data_management
} // namespace daal

#endif

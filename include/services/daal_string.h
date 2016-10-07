/** file daal_string.h */
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
//  Intel(R) DAAL string class.
//--
*/

#ifndef __DAAL_STRING__
#define __DAAL_STRING__

#include "base.h"

namespace daal
{
namespace services
{

namespace interface1
{
/**
 * @ingroup error_handling
 * @{
 */
/**
 * <a name="DAAL-CLASS-SERVICES__STRING"></a>
 * \brief Class that implements functionality of the string,
 *        an object that represents a sequence of characters
 */
class DAAL_EXPORT String : public Base
{
public:
    /**
     * Default constructor
     * \param[in] str       The sequence of characters that forms the string
     * \param[in] capacity  Number of characters that will be allocated to store the string
     */
    String(const char *str, size_t capacity = 0);

    /**
     * Copy constructor
     * \param[in] str       The sequence of characters that forms the string
     */
    String(const String &str);

    /**
     * Destructor
     */
    ~String();

    /**
     * Returns the number of characters in the string
     * \return The number of characters in the string
     */
    size_t length() const;

    /**
     * Extends the string by appending additional characters at the end of its current value
     * \param[in] str A string object whose values are copied at the end
     */
    void add(const String &str);

    /**
     * Extends the string by appending additional characters at the end of its current value
     * \param[in] str A string object whose values are copied at the end
     */
    String &operator+ (const String &str);

    /**
     * Returns the pointer to a character of the string
     * \param[in] index     Index of the character
     * \return  Pointer to the character of the string
     */
    char operator[] (size_t index) const;

    /**
     * Returns the pointer to a character of the string
     * \param[in] index     Index of the character
     * \return  Pointer to the character of the string
     */
    char get(size_t index) const;

    /**
     * Returns the content of the string as array of characters
     * \return The content of the string as array of characters
     */
    const char *c_str() const;

    static const int __DAAL_STR_MAX_SIZE;   /*!< Maximal length of the string */

private:
    char *_c_str;

    void initialize(const char *str, const size_t length);
};
/** @} */
} // namespace interface1
using interface1::String;

}
}
#endif

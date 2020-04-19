/* file: daal_level_zero_types.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef _DAAL_LEVEL_ZERO_TYPES
#define _DAAL_LEVEL_ZERO_TYPES

#if defined(__cplusplus)
extern "C"
{
#endif

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAKE_VERSION
    /// @brief Generates generic 'One API' API versions
    #define ZE_MAKE_VERSION(_major, _minor) ((_major << 16) | (_minor & 0x0000ffff))
#endif // ZE_MAKE_VERSION

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Handle of driver's device object
    typedef struct _ze_device_handle_t * ze_device_handle_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Handle of driver's module object
    typedef struct _ze_module_handle_t * ze_module_handle_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Handle of module's build log object
    typedef struct _ze_module_build_log_handle_t * ze_module_build_log_handle_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Defines Return/Error codes
    typedef enum _ze_result_t
    {
        ZE_RESULT_SUCCESS           = 0,                       ///< [Core] success
        ZE_RESULT_NOT_READY         = 1,                       ///< [Core] synchronization primitive not signaled
        ZE_RESULT_ERROR_DEVICE_LOST = 0x70000001,              ///< [Core] device hung, reset, was removed, or driver update occurred
        ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY,                    ///< [Core] insufficient host memory to satisfy call
        ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY,                  ///< [Core] insufficient device memory to satisfy call
        ZE_RESULT_ERROR_MODULE_BUILD_FAILURE,                  ///< [Core] error occurred when building module, see build log for details
        ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS = 0x70010000, ///< [Tools] access denied due to permission level
        ZE_RESULT_ERROR_NOT_AVAILABLE,                         ///< [Tools] resource already in use and simultaneous access not allowed
        ZE_RESULT_ERROR_UNINITIALIZED = 0x78000001,            ///< [Validation] driver is not initialized
        ZE_RESULT_ERROR_UNSUPPORTED_VERSION,                   ///< [Validation] generic error code for unsupported versions
        ZE_RESULT_ERROR_UNSUPPORTED_FEATURE,                   ///< [Validation] generic error code for unsupported features
        ZE_RESULT_ERROR_INVALID_ARGUMENT,                      ///< [Validation] generic error code for invalid arguments
        ZE_RESULT_ERROR_INVALID_NULL_HANDLE,                   ///< [Validation] handle argument is not valid
        ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE,                  ///< [Validation] object pointed to by handle still in-use by device
        ZE_RESULT_ERROR_INVALID_NULL_POINTER,                  ///< [Validation] pointer argument may not be nullptr
        ZE_RESULT_ERROR_INVALID_SIZE,                          ///< [Validation] size argument is invalid (e.g., must not be zero)
        ZE_RESULT_ERROR_UNSUPPORTED_SIZE,                      ///< [Validation] size argument is not supported by the device (e.g., too
                                                               ///< large)
        ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT,                 ///< [Validation] alignment argument is not supported by the device (e.g.,
                                                               ///< too small)
        ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT,        ///< [Validation] synchronization object in invalid state
        ZE_RESULT_ERROR_INVALID_ENUMERATION,                   ///< [Validation] enumerator argument is not valid
        ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION,               ///< [Validation] enumerator argument is not supported by the device
        ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT,              ///< [Validation] image format is not supported by the device
        ZE_RESULT_ERROR_INVALID_NATIVE_BINARY,                 ///< [Validation] native binary is not supported by the device
        ZE_RESULT_ERROR_INVALID_GLOBAL_NAME,                   ///< [Validation] global variable is not found in the module
        ZE_RESULT_ERROR_INVALID_KERNEL_NAME,                   ///< [Validation] kernel name is not found in the module
        ZE_RESULT_ERROR_INVALID_FUNCTION_NAME,                 ///< [Validation] function name is not found in the module
        ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION,          ///< [Validation] group size dimension is not valid for the kernel or
                                                               ///< device
        ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION,        ///< [Validation] global width dimension is not valid for the kernel or
                                                               ///< device
        ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX,         ///< [Validation] kernel argument index is not valid for kernel
        ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE,          ///< [Validation] kernel argument size does not match kernel
        ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE,        ///< [Validation] value of kernel attribute is not valid for the kernel or
                                                               ///< device
        ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE,             ///< [Validation] command list type does not match command queue type
        ZE_RESULT_ERROR_OVERLAPPING_REGIONS,                   ///< [Validation] copy operations do not support overlapping regions of
                                                               ///< memory
        ZE_RESULT_ERROR_UNKNOWN = 0x7fffffff,                  ///< [Core] unknown or internal error

    } ze_result_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief API version of ::ze_module_desc_t
    typedef enum _ze_module_desc_version_t
    {
        ZE_MODULE_DESC_VERSION_CURRENT = ZE_MAKE_VERSION(0, 91), ///< version 0.91

    } ze_module_desc_version_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Supported module creation input formats
    typedef enum _ze_module_format_t
    {
        ZE_MODULE_FORMAT_IL_SPIRV = 0, ///< Format is SPIRV IL format
        ZE_MODULE_FORMAT_NATIVE,       ///< Format is device native format

    } ze_module_format_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Specialization constants - User defined constants
    typedef struct _ze_module_constants_t
    {
        uint32_t numConstants;            ///< [in] Number of specialization constants.
        const uint32_t * pConstantIds;    ///< [in] Pointer to array of IDs that is sized to numConstants.
        const uint64_t * pConstantValues; ///< [in] Pointer to array of values that is sized to numConstants.

    } ze_module_constants_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Module descriptor
    typedef struct _ze_module_desc_t
    {
        ze_module_desc_version_t version;         ///< [in] ::ZE_MODULE_DESC_VERSION_CURRENT
        ze_module_format_t format;                ///< [in] Module format passed in with pInputModule
        size_t inputSize;                         ///< [in] size of input IL or ISA from pInputModule.
        const uint8_t * pInputModule;             ///< [in] pointer to IL or ISA
        const char * pBuildFlags;                 ///< [in] string containing compiler flags. See programming guide for build
                                                  ///< flags.
        const ze_module_constants_t * pConstants; ///< [in] pointer to specialization constants. Valid only for SPIR-V input.
                                                  ///< This must be set to nullptr if no specialization constants are
                                                  ///< provided.

    } ze_module_desc_t;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _DAAL_LEVEL_ZERO_TYPES

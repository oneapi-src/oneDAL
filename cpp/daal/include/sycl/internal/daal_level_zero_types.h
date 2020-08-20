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
    /// @brief Handle of a driver instance
    typedef struct _ze_driver_handle_t * ze_driver_handle_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Handle of driver's device object
    typedef struct _ze_device_handle_t * ze_device_handle_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Handle of driver's context object
    typedef struct _ze_context_handle_t * ze_context_handle_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Handle of driver's command queue object
    typedef struct _ze_command_queue_handle_t * ze_command_queue_handle_t;

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
        ZE_RESULT_SUCCESS                        = 0,                ///< [Core] success
        ZE_RESULT_NOT_READY                      = 1,                ///< [Core] synchronization primitive not signaled
        ZE_RESULT_ERROR_DEVICE_LOST              = 0x70000001,       ///< [Core] device hung, reset, was removed, or driver update occurred
        ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY       = 0x70000002,       ///< [Core] insufficient host memory to satisfy call
        ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY     = 0x70000003,       ///< [Core] insufficient device memory to satisfy call
        ZE_RESULT_ERROR_MODULE_BUILD_FAILURE     = 0x70000004,       ///< [Core] error occurred when building module, see build log for details
        ZE_RESULT_ERROR_MODULE_LINK_FAILURE      = 0x70000005,       ///< [Core] error occurred when linking modules, see build log for details
        ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS = 0x70010000,       ///< [Sysman] access denied due to permission level
        ZE_RESULT_ERROR_NOT_AVAILABLE            = 0x70010001,       ///< [Sysman] resource already in use and simultaneous access not allowed
                                                                     ///< or resource was removed
        ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE = 0x70020000,         ///< [Tools] external required dependency is unavailable or missing
        ZE_RESULT_ERROR_UNINITIALIZED          = 0x78000001,         ///< [Validation] driver is not initialized
        ZE_RESULT_ERROR_UNSUPPORTED_VERSION    = 0x78000002,         ///< [Validation] generic error code for unsupported versions
        ZE_RESULT_ERROR_UNSUPPORTED_FEATURE    = 0x78000003,         ///< [Validation] generic error code for unsupported features
        ZE_RESULT_ERROR_INVALID_ARGUMENT       = 0x78000004,         ///< [Validation] generic error code for invalid arguments
        ZE_RESULT_ERROR_INVALID_NULL_HANDLE    = 0x78000005,         ///< [Validation] handle argument is not valid
        ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE   = 0x78000006,         ///< [Validation] object pointed to by handle still in-use by device
        ZE_RESULT_ERROR_INVALID_NULL_POINTER   = 0x78000007,         ///< [Validation] pointer argument may not be nullptr
        ZE_RESULT_ERROR_INVALID_SIZE           = 0x78000008,         ///< [Validation] size argument is invalid (e.g., must not be zero)
        ZE_RESULT_ERROR_UNSUPPORTED_SIZE       = 0x78000009,         ///< [Validation] size argument is not supported by the device (e.g., too
                                                                     ///< large)
        ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT = 0x7800000a,          ///< [Validation] alignment argument is not supported by the device (e.g.,
                                                                     ///< too small)
        ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT = 0x7800000b, ///< [Validation] synchronization object in invalid state
        ZE_RESULT_ERROR_INVALID_ENUMERATION            = 0x7800000c, ///< [Validation] enumerator argument is not valid
        ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION        = 0x7800000d, ///< [Validation] enumerator argument is not supported by the device
        ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT       = 0x7800000e, ///< [Validation] image format is not supported by the device
        ZE_RESULT_ERROR_INVALID_NATIVE_BINARY          = 0x7800000f, ///< [Validation] native binary is not supported by the device
        ZE_RESULT_ERROR_INVALID_GLOBAL_NAME            = 0x78000010, ///< [Validation] global variable is not found in the module
        ZE_RESULT_ERROR_INVALID_KERNEL_NAME            = 0x78000011, ///< [Validation] kernel name is not found in the module
        ZE_RESULT_ERROR_INVALID_FUNCTION_NAME          = 0x78000012, ///< [Validation] function name is not found in the module
        ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION   = 0x78000013, ///< [Validation] group size dimension is not valid for the kernel or
                                                                     ///< device
        ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 0x78000014, ///< [Validation] global width dimension is not valid for the kernel or
                                                                     ///< device
        ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX  = 0x78000015, ///< [Validation] kernel argument index is not valid for kernel
        ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE   = 0x78000016, ///< [Validation] kernel argument size does not match kernel
        ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 0x78000017, ///< [Validation] value of kernel attribute is not valid for the kernel or
                                                                     ///< device
        ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED = 0x78000018,        ///< [Validation] module with imports needs to be linked before kernels can
                                                                     ///< be created from it.
        ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE = 0x78000019,      ///< [Validation] command list type does not match command queue type
        ZE_RESULT_ERROR_OVERLAPPING_REGIONS       = 0x7800001a,      ///< [Validation] copy operations do not support overlapping regions of
                                                                     ///< memory
        ZE_RESULT_ERROR_UNKNOWN = 0x7ffffffe,                        ///< [Core] unknown or internal error
        ZE_RESULT_FORCE_UINT32  = 0x7fffffff

    } ze_result_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Defines structure types
    typedef enum _ze_structure_type_t
    {
        ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES                 = 0x1,        ///< ::ze_driver_properties_t
        ZE_STRUCTURE_TYPE_DRIVER_IPC_PROPERTIES             = 0x2,        ///< ::ze_driver_ipc_properties_t
        ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES                 = 0x3,        ///< ::ze_device_properties_t
        ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES         = 0x4,        ///< ::ze_device_compute_properties_t
        ZE_STRUCTURE_TYPE_DEVICE_MODULE_PROPERTIES          = 0x5,        ///< ::ze_device_module_properties_t
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES    = 0x6,        ///< ::ze_command_queue_group_properties_t
        ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES          = 0x7,        ///< ::ze_device_memory_properties_t
        ZE_STRUCTURE_TYPE_DEVICE_MEMORY_ACCESS_PROPERTIES   = 0x8,        ///< ::ze_device_memory_access_properties_t
        ZE_STRUCTURE_TYPE_DEVICE_CACHE_PROPERTIES           = 0x9,        ///< ::ze_device_cache_properties_t
        ZE_STRUCTURE_TYPE_DEVICE_IMAGE_PROPERTIES           = 0xa,        ///< ::ze_device_image_properties_t
        ZE_STRUCTURE_TYPE_DEVICE_P2P_PROPERTIES             = 0xb,        ///< ::ze_device_p2p_properties_t
        ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES = 0xc,        ///< ::ze_device_external_memory_properties_t
        ZE_STRUCTURE_TYPE_CONTEXT_DESC                      = 0xd,        ///< ::ze_context_desc_t
        ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC                = 0xe,        ///< ::ze_command_queue_desc_t
        ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC                 = 0xf,        ///< ::ze_command_list_desc_t
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC                   = 0x10,       ///< ::ze_event_pool_desc_t
        ZE_STRUCTURE_TYPE_EVENT_DESC                        = 0x11,       ///< ::ze_event_desc_t
        ZE_STRUCTURE_TYPE_FENCE_DESC                        = 0x12,       ///< ::ze_fence_desc_t
        ZE_STRUCTURE_TYPE_IMAGE_DESC                        = 0x13,       ///< ::ze_image_desc_t
        ZE_STRUCTURE_TYPE_IMAGE_PROPERTIES                  = 0x14,       ///< ::ze_image_properties_t
        ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC             = 0x15,       ///< ::ze_device_mem_alloc_desc_t
        ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC               = 0x16,       ///< ::ze_host_mem_alloc_desc_t
        ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES      = 0x17,       ///< ::ze_memory_allocation_properties_t
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC       = 0x18,       ///< ::ze_external_memory_export_desc_t
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD         = 0x19,       ///< ::ze_external_memory_import_fd_t
        ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD         = 0x1a,       ///< ::ze_external_memory_export_fd_t
        ZE_STRUCTURE_TYPE_MODULE_DESC                       = 0x1b,       ///< ::ze_module_desc_t
        ZE_STRUCTURE_TYPE_MODULE_PROPERTIES                 = 0x1c,       ///< ::ze_module_properties_t
        ZE_STRUCTURE_TYPE_KERNEL_DESC                       = 0x1d,       ///< ::ze_kernel_desc_t
        ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES                 = 0x1e,       ///< ::ze_kernel_properties_t
        ZE_STRUCTURE_TYPE_SAMPLER_DESC                      = 0x1f,       ///< ::ze_sampler_desc_t
        ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC                 = 0x20,       ///< ::ze_physical_mem_desc_t
        ZE_STRUCTURE_TYPE_RAYTRACING_MEM_ALLOC_EXT_DESC     = 0x00010001, ///< ::ze_raytracing_mem_alloc_ext_desc_t
        ZE_STRUCTURE_TYPE_FORCE_UINT32                      = 0x7fffffff

    } ze_structure_type_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Supported module creation input formats
    typedef enum _ze_module_format_t
    {
        ZE_MODULE_FORMAT_IL_SPIRV     = 0, ///< Format is SPIRV IL format
        ZE_MODULE_FORMAT_NATIVE       = 1, ///< Format is device native format
        ZE_MODULE_FORMAT_FORCE_UINT32 = 0x7fffffff

    } ze_module_format_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Specialization constants - User defined constants
    typedef struct _ze_module_constants_t
    {
        uint32_t numConstants;         ///< [in] Number of specialization constants.
        const uint32_t * pConstantIds; ///< [in][range(0, numConstants)] Array of IDs that is sized to
                                       ///< numConstants.
        const void ** pConstantValues; ///< [in][range(0, numConstants)] Array of pointers to values that is sized
                                       ///< to numConstants.

    } ze_module_constants_t;

    ///////////////////////////////////////////////////////////////////////////////
    /// @brief Module descriptor
    typedef struct _ze_module_desc_t
    {
        ze_structure_type_t stype;                ///< [in] type of this structure
        const void * pNext;                       ///< [in][optional] pointer to extension-specific structure
        ze_module_format_t format;                ///< [in] Module format passed in with pInputModule
        size_t inputSize;                         ///< [in] size of input IL or ISA from pInputModule.
        const uint8_t * pInputModule;             ///< [in] pointer to IL or ISA
        const char * pBuildFlags;                 ///< [in][optional] string containing compiler flags. Following options are supported.
                                                  ///<  - "-ze-opt-disable"
                                                  ///<       - Disable optimizations
                                                  ///<  - "-ze-opt-greater-than-4GB-buffer-required"
                                                  ///<       - Use 64-bit offset calculations for buffers.
                                                  ///<  - "-ze-opt-large-register-file"
                                                  ///<       - Increase number of registers available to threads.
                                                  ///<  - "-ze-opt-has-buffer-offset-arg"
                                                  ///<       - Extend stateless to stateful optimization to more
                                                  ///<         cases with the use of additional offset (e.g. 64-bit
                                                  ///<         pointer to binding table with 32-bit offset).
                                                  ///<  - "-g"
                                                  ///<       - Include debugging information.
        const ze_module_constants_t * pConstants; ///< [in][optional] pointer to specialization constants. Valid only for
                                                  ///< SPIR-V input. This must be set to nullptr if no specialization
                                                  ///< constants are provided.

    } ze_module_desc_t;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _DAAL_LEVEL_ZERO_TYPES

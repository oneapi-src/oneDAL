/* file: ZlibDecompressor.java */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

/**
 * @ingroup data_compression
 * @{
 */
package com.intel.daal.data_management.compression.zlib;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.compression.CompressionMethod;
import com.intel.daal.data_management.compression.Decompressor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__ZLIB__ZLIBDECOMPRESSOR"></a>
 *
 * @brief Specialization of the Decompressor class for ZLIB decompression method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * @par References
 *      - @ref ZlibCompressionParameter class
 */
public class ZlibDecompressor extends Decompressor {
    public ZlibCompressionParameter parameter; /*!< ZLIB compression parameters */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the ZLIB decompression algorithm
     * @param context   Context to manage the ZLIB decompression algorithm
     */
    public ZlibDecompressor(DaalContext context) {
        super(context, CompressionMethod.zlib);
        parameter = new ZlibCompressionParameter(context,
                cInitParameter(this.cObject, CompressionMethod.zlib.getValue()));
    }
}
/** @} */

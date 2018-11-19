/* file: ZlibCompressor.java */
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

/**
 * @ingroup data_compression
 * @{
 */
package com.intel.daal.data_management.compression.zlib;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.compression.CompressionMethod;
import com.intel.daal.data_management.compression.Compressor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__ZLIB__ZLIBCOMPRESSOR"></a>
 *
 * @brief Specialization of the Compressor class for ZLIB compression method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * @par References
 *      - @ref ZlibCompressionParameter class
 */
public class ZlibCompressor extends Compressor {
    public ZlibCompressionParameter parameter; /*!< Zlib compression parameters */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the ZLIB compression algorithm
     * @param context   Context to manage the ZLIB compression algorithm
     */
    public ZlibCompressor(DaalContext context) {
        super(context, CompressionMethod.zlib);
        parameter = new ZlibCompressionParameter(context,
                cInitParameter(this.cObject, CompressionMethod.zlib.getValue()));
    }
}
/** @} */

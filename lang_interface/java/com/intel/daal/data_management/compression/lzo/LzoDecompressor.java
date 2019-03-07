/* file: LzoDecompressor.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
package com.intel.daal.data_management.compression.lzo;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.compression.CompressionMethod;
import com.intel.daal.data_management.compression.Decompressor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__LZO__LZODECOMPRESSOR"></a>
 *
 * @brief Specialization of the Decompressor class for the LZO decompression method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * @par References
 *      - @ref LzoCompressionParameter class
 */
public class LzoDecompressor extends Decompressor {
    public LzoCompressionParameter parameter; /*!< LZO compression parameters */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the LZO decompression algorithm
     * @param context   Context to manage the LZO decompression algorithm
     */
    public LzoDecompressor(DaalContext context) {
        super(context, CompressionMethod.lzo);
        parameter = new LzoCompressionParameter(context,
                cInitParameter(this.cObject, CompressionMethod.lzo.getValue()));
    }
}
/** @} */

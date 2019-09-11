/* file: RleCompressor.java */
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
package com.intel.daal.data_management.compression.rle;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.compression.CompressionMethod;
import com.intel.daal.data_management.compression.Compressor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__RLE__RLECOMPRESSOR"></a>
 *
 * @brief Specialization of the Compressor class for the RLE compression method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * @par References
 *      - @ref RleCompressionParameter class
 */
public class RleCompressor extends Compressor {
    public RleCompressionParameter parameter; /*!< RLE compression parameters */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the RLE compression algorithm
     * @param context   Context to manage the RLE compression algorithm
     */
    public RleCompressor(DaalContext context) {
        super(context, CompressionMethod.rle);
        parameter = new RleCompressionParameter(context,
                cInitParameter(this.cObject, CompressionMethod.rle.getValue()));
    }
}
/** @} */

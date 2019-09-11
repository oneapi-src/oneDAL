/* file: Bzip2Decompressor.java */
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
package com.intel.daal.data_management.compression.bzip2;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.compression.CompressionMethod;
import com.intel.daal.data_management.compression.Decompressor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__BZIP2__BZIP2DECOMPRESSOR"></a>
 *
 * @brief Implementation of the Decompressor class for the BZIP2 decompression method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * @par References
 *      - @ref Bzip2CompressionParameter class
 */
public class Bzip2Decompressor extends Decompressor {
    public Bzip2CompressionParameter parameter; /*!< Bzip2 compression parameters */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the BZIP2 decompression algorithm
     * @param context   Context to manage the BZIP2 decompression algorithm
     */
    public Bzip2Decompressor(DaalContext context) {
        super(context, CompressionMethod.bzip2);
        parameter = new Bzip2CompressionParameter(context,
                cInitParameter(this.cObject, CompressionMethod.bzip2.getValue()));
    }
}
/** @} */

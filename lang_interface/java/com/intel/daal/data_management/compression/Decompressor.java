/* file: Decompressor.java */
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
package com.intel.daal.data_management.compression;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__DECOMPRESSOR"></a>
 * @brief The base class that provides methods for the decompression
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * @par References
 *      - @ref CompressionMethod class
 */
public class Decompressor extends Compression {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Decompressor constructor
     * @param context Context to manage decompressor
     * @param method  Compression method, @ref CompressionMethod
     */
    public Decompressor(DaalContext context, CompressionMethod method) {
        super(context);
        if (method != CompressionMethod.zlib && method != CompressionMethod.lzo && method != CompressionMethod.rle
                && method != CompressionMethod.bzip2) {
            throw new IllegalArgumentException("method unsupported");
        }
        this.cObject = cInit(method.getValue());
    }

    private native long cInit(int comprMethod);

    protected native long cInitParameter(long comprAddr, int comprMethod);
}
/** @} */

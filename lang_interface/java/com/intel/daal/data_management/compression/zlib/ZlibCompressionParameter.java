/* file: ZlibCompressionParameter.java */
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
import com.intel.daal.data_management.compression.CompressionParameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__ZLIB__ZLIBCOMPRESSIONPARAMETER"></a>
 *
 * @brief Parameter for ZLIB compression and decompression
 *
 * @par Enumerations
 *      - @ref CompressionLevel - %Compression levels enumeration
 */
public class ZlibCompressionParameter extends CompressionParameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
    * ZlibCompressionParameter constructor
    */
    public ZlibCompressionParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Sets optional GZIP header flag
     *  @param gzHeader Optional GZIP header flag. True if simple GZIP header is included, false otherwise
     */
    public void setGzHeader(boolean gzHeader) {
        cSetGzHeader(this.cObject, gzHeader);
    }

    /**
     *  Returns optional GZIP header flag
     *  @return Optional GZIP header flag. True if simple GZIP header is included, false otherwise
     */
    public boolean getGzHeader() {
        return cGetGzHeader(this.cObject);
    }

    private native void cSetGzHeader(long parAddr, boolean gzHeader);

    private native boolean cGetGzHeader(long parAddr);
}
/** @} */

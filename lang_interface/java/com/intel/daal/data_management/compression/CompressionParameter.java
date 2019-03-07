/* file: CompressionParameter.java */
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
import com.intel.daal.algorithms.Parameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__COMPRESSIONPARAMETER"></a>
 * @brief Parameters for the compression and decompression
 *
 * @par References
 *      - @ref CompressionLevel - Compression levels
 */
public class CompressionParameter extends Parameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
    * CompressionParameter constructor
    */
    public CompressionParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     * Sets the compression level
     * @param level   Compression level
     */
    public void setCompressionLevel(CompressionLevel level) {
        cSetCompressionLevel(this.cObject, level.getValue());
    }

    /**
     * Returns the compression level
     * @return Compression level
     */
    public CompressionLevel getCompressionLevel() {
        CompressionLevel cLevel = new CompressionLevel(cGetCompressionLevel(this.cObject));
        return cLevel;
    }

    private native void cSetCompressionLevel(long parAddr, int cLevel);

    private native int cGetCompressionLevel(long parAddr);
}
/** @} */

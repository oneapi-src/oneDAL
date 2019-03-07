/* file: Bzip2CompressionParameter.java */
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
import com.intel.daal.data_management.compression.CompressionParameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__BZIP2__BZIP2COMPRESSIONPARAMETER"></a>
 *
 * @brief Parameter for the BZIP2 compression and decompression,
 *                 CompressionLevel.defaultLevel is equal to BZIP2 compression level 9
 *
 * @par Enumerations
 *      - @ref CompressionLevel - %Compression levels enumeration
 */
public class Bzip2CompressionParameter extends CompressionParameter {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
    * Bzip2CompressionParameter constructor
    */
    public Bzip2CompressionParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

}
/** @} */

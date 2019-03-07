/* file: CompressionMethod.java */
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

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__COMPRESSIONMETHOD"></a>
 * @brief Compression and decompression methods
 */
public final class CompressionMethod {
    private int _value;

    /**
     * Constructs the compression method object using the provided value
     * @param value     Value corresponding to the compression method object
     */
    public CompressionMethod(int value) {
        _value = value;
    }

    /**
     * Returns the value corresponding to the compression method object
     * @return Value corresponding to the compression method object
     */
    public int getValue() {
        return _value;
    }

    private static final int Zlib  = 0;
    private static final int Lzo   = 1;
    private static final int Rle   = 2;
    private static final int Bzip2 = 3;

    public static final CompressionMethod zlib  = new CompressionMethod(Zlib);  /*!< DEFLATE compression method with ZLIB block header
                                                                                     or simple GZIP block header */
    public static final CompressionMethod lzo   = new CompressionMethod(
            Lzo);                                                               /*!< LZO1X compatible compression method */
    public static final CompressionMethod rle   = new CompressionMethod(Rle);   /*!< Run-Length Encoding method */
    public static final CompressionMethod bzip2 = new CompressionMethod(Bzip2); /*!< BZIP2 compression method */
}
/** @} */

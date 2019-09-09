/* file: CompressionMethod.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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
package com.intel.daal.data_management.compression;

import java.lang.annotation.Native;

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

    @Native private static final int Zlib  = 0;
    @Native private static final int Lzo   = 1;
    @Native private static final int Rle   = 2;
    @Native private static final int Bzip2 = 3;

    public static final CompressionMethod zlib  = new CompressionMethod(Zlib);  /*!< DEFLATE compression method with ZLIB block header
                                                                                     or simple GZIP block header */
    public static final CompressionMethod lzo   = new CompressionMethod(
            Lzo);                                                               /*!< LZO1X compatible compression method */
    public static final CompressionMethod rle   = new CompressionMethod(Rle);   /*!< Run-Length Encoding method */
    public static final CompressionMethod bzip2 = new CompressionMethod(Bzip2); /*!< BZIP2 compression method */
}
/** @} */

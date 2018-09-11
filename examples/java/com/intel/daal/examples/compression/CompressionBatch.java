/* file: CompressionBatch.java */
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

/*
 //  Content:
 //     Java example of compression in the batch processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-COMPRESSIONBATCH">
 * @example CompressionBatch.java
 */

package com.intel.daal.examples.compression;

import java.io.RandomAccessFile;
import java.util.zip.CRC32;
import java.util.zip.Checksum;

import com.intel.daal.data_management.compression.*;
import com.intel.daal.data_management.compression.zlib.*;
import com.intel.daal.services.DaalContext;

class CompressionBatch {

    private static final String dataset = "../data/batch/logitboost_train.csv";

    private static byte[] rawData;          /* Data to compress */
    private static byte[] compressedData;   /* Result of compression */
    private static byte[] decompressedData; /* Result of decompression */

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Read data from a file and allocate memory */
        prepareMemory();

        /* Create a compressor */
        ZlibCompressor compressor = new ZlibCompressor(context);
        compressor.parameter.setCompressionLevel(CompressionLevel.Level9);
        compressor.parameter.setGzHeader(true);

        /* Create a stream for compression */
        CompressionStream compressionStream = new CompressionStream(context, compressor);

        /* Write raw data to the compression stream and compress if needed */
        compressionStream.add(rawData);

        /* Get the size of the compressed data */
        compressedData = new byte[(int) compressionStream.getCompressedDataSize()];

        /* Store the compressed data */
        compressionStream.copyCompressedArray(compressedData);

        /* Create a decompressor */
        ZlibDecompressor decompressor = new ZlibDecompressor(context);
        decompressor.parameter.setGzHeader(true);

        /* Create a stream for decompression */
        DecompressionStream decompressionStream = new DecompressionStream(context, decompressor);

        /* Write the compressed data to the decompression stream and decompress it */
        decompressionStream.add(compressedData);

        /* Get the size of the decompressed data */
        decompressedData = new byte[(int) decompressionStream.getDecompressedDataSize()];

        /* Store the decompressed data */
        decompressionStream.copyDecompressedArray(decompressedData);

        /* Compute and print checksums for rawData and decompressedData */
        printCRC32(rawData, decompressedData);

        context.dispose();
    }

    private static void prepareMemory() throws java.io.FileNotFoundException, java.io.IOException {

        /* Read data from a file */
        rawData = readData();
    }

    private static byte[] readData() throws java.io.FileNotFoundException, java.io.IOException {

        /* Read an input file */
        RandomAccessFile file = new RandomAccessFile(dataset, "r");
        int dataLength = (int) file.length();
        byte[] sentData = new byte[dataLength];
        file.read(sentData);
        file.close();

        return sentData;
    }

    private static void printCRC32(byte[] sentData, byte[] receivedData) {

        /* Compute checksums for full input data and full received data */
        Checksum crcSentDataStream = new CRC32();
        crcSentDataStream.update(sentData, 0, sentData.length);
        Checksum crcReceivedDataStream = new CRC32();
        crcReceivedDataStream.update(receivedData, 0, receivedData.length);

        System.out.println("\nCompression example program results:\n");

        System.out.println("Raw data checksum:    0x" + Integer.toHexString((int) crcSentDataStream.getValue()));
        System.out.println("Decompressed data checksum: 0x" + Integer.toHexString((int) crcReceivedDataStream.getValue()));

        if (sentData.length != receivedData.length) {
            System.out.println("ERROR: Received data size (" + receivedData.length
                    + ") mismatches with the sent data size (" + sentData.length + ")");
        } else if (crcSentDataStream.getValue() != crcReceivedDataStream.getValue()) {
            System.out.println("ERROR: Received data CRC mismatches with the sent data CRC");
        } else {
            System.out.println("OK: Decompressed data CRC matches with the raw data CRC");
        }
    }
}

# file: compression_online.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#===============================================================================

#
# !  Content:
# !    Python example of compression in the online processing mode
# !
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-COMPRESSION_ONLINE"></a>
## \example compression_online.py
#

import os
import sys

if sys.version[0] == '2':
    import Queue as Queue
else:
    import queue as Queue

import numpy as np

from daal.data_management import Compressor_Zlib, Decompressor_Zlib, level9, DecompressionStream, CompressionStream

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import getCRC32, readTextFile

datasetFileName = os.path.join('..', 'data', 'online', 'logitboost_train.csv')

# Queue for sending and receiving compressed data blocks
# queue_DataBlock
sendReceiveQueue = Queue.Queue()

maxDataBlockSize = 16384     # Maximum size of a data block
userDefinedBlockSize = 7000  # Size for read data from a decompression stream

def getDataBlock(sentDataStream, availableDataSize):
    cur_pos = sentDataStream.size - availableDataSize

    # return next slice of array and remaining datasize
    if availableDataSize >= maxDataBlockSize:
        return (sentDataStream[cur_pos:cur_pos + maxDataBlockSize], availableDataSize - maxDataBlockSize)
    elif availableDataSize < maxDataBlockSize and availableDataSize > 0:
        return (sentDataStream[cur_pos:cur_pos + availableDataSize], 0)
    return (None,None)


def sendDataBlock(block):
    currentBlock = np.copy(block)
    # Push the current compressed block to the queue
    sendReceiveQueue.put(currentBlock)


def receiveDataBlock():
    # Stop at the end of the queue
    if sendReceiveQueue.empty():
        return None
    # Receive and copy the current compressed block from the queue
    return np.copy(sendReceiveQueue.get())


def printCRC32(sentDataStream, receivedDataStream):
    # Compute checksums for full input data and full received data
    crcSentDataStream = getCRC32(sentDataStream)
    crcReceivedDataStream = getCRC32(receivedDataStream)

    print("\nCompression example program results:\n")

    print("Input data checksum:    0x{:02X}".format(crcSentDataStream))
    print("Received data checksum: 0x{:02X}".format(crcReceivedDataStream))

    if sentDataStream.size != receivedDataStream.size:
        print("ERROR: Received data size mismatches with the sent data size")

    elif crcSentDataStream != crcReceivedDataStream:
        print("ERROR: Received data CRC mismatches with the sent data CRC")
    else:
        print("OK: Received data CRC matches with the sent data CRC")


if __name__ == "__main__":
    # Read data from a file and allocate memory
    sentDataStream = readTextFile(datasetFileName)

    # Create a compressor
    compressor = Compressor_Zlib()
    compressor.parameter.gzHeader = True
    compressor.parameter.level = level9

    # Create a stream for compression
    compressionStream = CompressionStream(compressor)

    # Receive data by blocks from sentDataStream for further compression and send it
    (uncompressedDataBlock, availableDataSize) = getDataBlock(sentDataStream, sentDataStream.size)
    while uncompressedDataBlock is not None:
        # Put a data block to compressionStream and compress if needed
        compressionStream.push_back(uncompressedDataBlock)

        # Get access to compressed blocks stored in compressionStream without an actual compressed data copy
        compressedBlocks = compressionStream.getCompressedBlocksCollection()

        # Send compressed blocks stored in compressionStream
        for i in range(compressedBlocks.size()):
            # Send the current compressed block from compressionStream
            sendDataBlock(compressedBlocks[i].getArray())

        # Receive the next data block for compression
        (uncompressedDataBlock, availableDataSize) = getDataBlock(sentDataStream, availableDataSize)

    # Create a decompressor
    decompressor = Decompressor_Zlib()
    decompressor.parameter.gzHeader = True

    # Create a stream for decompression
    decompressionStream = DecompressionStream(decompressor)

    # Actual size of decompressed data currently read from decompressionStream
    readSize = 0

    # Received uncompressed data stream
    receivedDataStream = np.empty(0, dtype=np.uint8)
    tmp_block = np.empty(userDefinedBlockSize, dtype=np.uint8)

    # Receive compressed data by blocks
    compressedDataBlock = receiveDataBlock()

    while compressedDataBlock is not None:
        # Write a received block to decompressionStream
        decompressionStream.push_back(compressedDataBlock)

        # Asynchronous read from decompressionStream
        while True:
            # Read userDefinedBlockSize bytes from decompressionStream to the end of receivedDataStream
            readSize = decompressionStream.copyDecompressedArray(tmp_block)
            if readSize == 0:
                break
            # Update the actual data size in receivedDataStream
            receivedDataStream = np.concatenate((receivedDataStream, tmp_block[:readSize]))

        # Receive next block
        compressedDataBlock = receiveDataBlock()

    # Compute and print checksums for sentDataStream and receivedDataStream
    printCRC32(sentDataStream, receivedDataStream)

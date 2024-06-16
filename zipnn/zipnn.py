import time
import sys
import os
import math
import numpy as np
import zstandard as zstd
from zipnn.util_header import EnumMethod, EnumLossy
import split_dtype

class ZipNN:

    def __init__(
        self,
        method: str = "zstd",
        input_format: str = "byte",
        bytearray_dtype: str = "float32",
        is_monotonic: bool = False, 
        header_lean : bool = False,

        bg: int = 0,
        max_threads: int = 0,
        bg_compression_threshold=0.95,
        reorder_signbit: int = 2,

        delta_compressed_type: str = None,


        lossy_compressed_type=None,
        lossy_compressed_factor=27,

        is_streaming: bool = False,
        streaming_chunk_kb: int = 64 * 1024,

        input_file: str = "byte",
        compressed_file: str = None,
        decompressed_file: str = None,

        zstd_level: int = 3,
        lz4_compression_level: int = 0,
    ):
        """
        Zipnn class is used to compress and decompress data in byte, file, and Torch tensor formats.
        Additionally, there is support for byte grouping, lossy compression, and delta compression,
            allowing you to use some, all, or none of these techniques.

        Parameters
        -------------------------------------
        method: string
                Chosen compression method. The options are: ‘zstd’/’ZSTD’, ‘lz4’/’LZ4’, ‘snappy’/’SNAPPY’.
                Default is ‘zstd’.

        input_format: string
                The type of the input, the same will be for the output.
                The options are ‘byte’, ‘torch’, ‘numpy’. -> ‘file’ is not implemented yet
                Default is ‘byte’.

          
        bytearray_dtype: string,
                Chosen dtype for bytearray: The options are: ‘float32’, ‘uint32’, ‘uint16', 'bfloat16', 'float16'
                Default is ‘float32’.

        is_monotonic : bool, 
                The  dataset is monotonic.
                Default is ‘False’.

         header_lean : bool,
                The header is lean, no need to save the dtype and the shape
                Default is ‘False’.

        bg: int
                Number of partitions for byte grouping.
                If set to zero, decide according to the dtype
                If set to 1 - no byte grouping, 1 group running vanilla compression method.
                If 2,4 - Byte group to 2 groups or 4 roups respectively
        
        max_threads: int 
                The maximum threads for th ecompression and the byte/bit reorder.
                If 0, the code decide according to the dataset len

        bg_compression_threshold: float
                Compression threshhold for byte grouping.
                Only relevant for a compression that uses byte grouping.
                Default is 0.95.

        reorder_signbit: int
                This reorder the bits of the float32 or bfloat16 to better compression.
                It puts the exponent first than the sign bit and than the mantissa.
                For float32 the value should be 32
                For bfloat16 the value should be 16
                Default is 0 (don't reorder)


       delta_compressed_type: string
              NOT IMPLEMENTED YET.
               Type for delta compression.
               Options are 'byte', 'file'.
               Default is ‘None’.


        lossy_compressed_type: string
                Type for lossy compression.
                Supporting only 'integer' ('unsigned' in the future).
                Only relevant if compression is lossy.
                Defaul is None.

        lossy_compressed_factor: int
                Compression factor for lossy compression.
                O nly relevant if compression is lossy.
                Default is 27.

        is_streaming: bool
                NOT IMPLEMENTED YET.
                If true – signals compression is for a stream of data.
                Deafult is False.

        streaming_chunk_kb: int
                Chunk size for streaming.
                Only relevant if is_steaming is True.
                Default is 1MB.

        input_file: string
                Path to the input file.
                If ‘file’ is the input type – enter file name.
                Default is ‘byte’.


        compressed_file: string
                Path to the compressed file.
                Only relevant if compressed_ret_type is ‘file’.
                Default is None.

        decompressed_file: string
                Path to the decompressed file.
                Only relevant if compressed_ret_type is ‘file’.
                Defaul is None.

        zstd_level: int
                Compression level for ‘zstd’ compression.
                Only relevant if method is ‘zstd’.
                Default is 3.

        lz4_compression_level: int
                Compression level for ‘lz4’ compression.
                Only relevant if method is ‘lz4’.
                Deafult is 0.

        Returns
        -------------------------------------
        ZipNN class instance supporting a specific compression and decompression based on the input given.
        """

        self.method = EnumMethod(method).value
        self.input_format = input_format
        self.bytearray_dtype = bytearray_dtype,
        self.header_lean = header_lean,
        self.is_monotonic= is_monotonic, 


        self.delta_compressed_type = delta_compressed_type
        self.bg = bg
        self.bg_compression_threshold = bg_compression_threshold
        self.reorder_signbit = reorder_signbit
        self.lossy_compressed_type = EnumLossy.NONE if lossy_compressed_type is None else EnumLossy(lossy_compressed_type)
        self.lossy_compressed_factor = lossy_compressed_factor

        self.is_streaming = is_streaming
        self.streaming_chunk_kb = streaming_chunk_kb

        self.input_file = input_file
        self.compressed_file = compressed_file
        self.decompressed_file = decompressed_file

        self.lz4_compression_level = lz4_compression_level

        self._version_major = 0
        self._version_minor = 1
        self._version_tiny = 1
        self._import_dependencies(zstd_level, max_threads)
        self._header = bytearray(16)
        self._is_int = 0
        self._update_header()
        return None

    def _import_dependencies(self, zstd_level, max_threads):
        """
        Importing needed dependencies, based on the ZipNN compression method.

        Parameters
        -------------------------------------
        torch_dtype: string
                If torch_dtype isn't None, then torch needs to be imported for the decompression.
         zstd_level: int
                Compression level for ‘zstd’ compression.

        max_threads: int
                Number of threads to be used for ‘zstd’ compression.

        Returns
        -------------------------------------
        None.
        """
        if self.method == EnumMethod.ZSTD.value:
            self._zstd_compress = zstd.ZstdCompressor(level=zstd_level, threads=max_threads)
            self._zstd_decompress = zstd.ZstdDecompressor()

        elif self.method == EnumMethod.LZ4.value:
            try:
                global lz4
                import lz4.frame
            except ImportError:
                raise ImportError("LZ4 library is not installed. Please install it to use LZ4 compression.")

        elif self.method == EnumMethod.SNAPPY.value:
            try:
                global snappy
                import snappy
            except ImportError:
                raise ImportError("Snappy library is not installed. Please install it to use Snappy compression.")

        else:
            raise ValueError("Unsupported compression method")

        if self.input_format == "torch":
            global torch
            import torch

            global ZipNNTorchDtypeEnum, zipnn_get_dtype_bits, zipnn_multiply_if_max_below, zipnn_divide_int, zipnn_pack_shape, zipnn_unpack_shape
            from zipnn.util_torch import (
                ZipNNTorchDtypeEnum,
                zipnn_multiply_if_max_below,
                zipnn_get_dtype_bits,
                zipnn_divide_int,
                zipnn_pack_shape,
                zipnn_unpack_shape,
            )

        if self.lossy_compressed_type != EnumLossy.NONE:
            if self.input_format != "torch":
                raise ValueError("When use lossy compression the input have to be torch.tensor")

    def use_var(self, data, class_var):
        """
        Used to update ZipNN attributes. Updates to data if it isn't null, or to the ZipNN class default if it is.

        Parameters
        -------------------------------------
        data:
                data to update some ZipNN attribute.

        Returns
        -------------------------------------
        data if not None, or the class default value.
        """
        if data is not None:
            return data
        return class_var

    # Header: 16 Bytes
    # [0:1] 2 Bytes [ZN]
    # [2:4] 2 Bytes [Versions]
    # [5] Byte [method]
    # [6] Byte [delta_compressed] - True/ False
    # [7] Byte [bg]
    # [8] Byte [reorder_signbit] -
    # [9] Byte [torch.dtype] -
    # [10] Byte [lossy_compress_type]
    # [11] Byte [lossy_compress_factor]
    # [12] Byte [is_int]
    # [13] Byte [is_streming]
    # [14:15] 2 Bytes [streaming_chunk_kb]

    def _update_header_lossy(self, lossy_type, lossy_factor, is_int):
        """
        Updates header with values of lossy compression.

        Parameters
        -------------------------------------
        lossy_type: string
                ZipNN attribute lossy_compressed_type.

        lossy_factor: int
                ZipNN attribute lossy_compressed_factor.

        is_int: bool
                Flag value, True if tensor was modified in the lossy compression.

        Returns
        -------------------------------------
        None.
        """
        self._header[10] = lossy_type.value
        self._header[11] = lossy_factor
        self._header[12] = is_int

#    def _update_header_dtype(self):
#        """
#        Updates header with dtype if it's torch decompression.
#
#        Parameters
#        -------------------------------------
#        lossy_type: string
#                torch_dtype if any was used.
#                Default is None.
#
#        Returns
#        -------------------------------------
#        None.

    def _update_header(self, lossy_compressed_type=None, lossy_compressed_factor=None):
        """
        Updates header with dtype if it's torch decompression.

        Parameters
        -------------------------------------
        lossy_compressed_type: string
                ZipNN attribute lossy_compressed_type.
                Default is None.

        lossy_compressed_factor: int
                ZipNN attribute lossy_compressed_factor.
                Default is None.

        Returns
        -------------------------------------
        None.
        """
        self._header[0:2] = "ZN".encode("ascii")  # header ZN
        self._header[2] = self._version_major
        self._header[3] = self._version_minor
        self._header[4] = self._version_tiny
        self._header[5] = self.method
        self._header[6] = 1 if self.delta_compressed_type is not None else 0
        self._header[7] = 4 #; int(math.log2(self.bg))  # log 2
        self._header[8] = self.reorder_signbit
    #   self._update_header_dtype()  # self._header[8]

    #        self._header[9] = 0 # lossy_compressed_factor
    #        self._header[10] = 0 # is_int
    #        self._header[11] = 0 # streming no implemented yet
    #        self._header[12:13] = 0 # streming no implemented yet
    #        self._header[14:15] = 0 # reserved

    def _retrieve_header_dtype(self, num):
        """
        Retrieves dtype from header.

        Parameters
        -------------------------------------
        num: int
                Value of torch_dtype in header.

        Returns
        -------------------------------------
        Torch dtype
        """
        if num == 0:
            return None
        return ZipNNTorchDtypeEnum.from_code(num).dtype

    def _retrieve_header(self, ba_compress):
        """
        Retrieves header values, and returns header length.

        Parameters
        -------------------------------------
        ba_compress: byte
                Header data compressed to byte array.

        Returns
        -------------------------------------
        The header length.
        """
        mv = ba_compress
        header = mv[0 : len(self._header)]
        header_length = len(self._header)
        if header[0:2].decode("ascii") != "ZN":
            sys.exit("Header should start with ZN")
        self.version_major = header[2]
        self.version_minor = header[3]
        self.version_tiny = header[4]
        self.method = int(header[5])
        self.delta_compressed_type = int(header[6])
        self.bg = 2 ** int(header[7])
        self.reorder_signbit = int(header[8])
        self.torch_dtype = self._retrieve_header_dtype(int(header[9]))
        self.lossy_compressed_type = int(header[10])
        self.lossy_compressed_factor = int(header[11])
        self._is_int = int(header[12])
        if self.input_format == "torch":
            self.shape_bytes, shape_size = zipnn_unpack_shape(mv[len(self._header) :])
            header_length += shape_size
        return header_length

    #################
    ## compression ##
    #################

    def compress(
        self, data, compress_cpu_gpu="cpu", delta_second_data=None, lossy_compressed_type: str = None, lossy_compressed_factor: int = None
    ):  # the data/ delta_second_data is "byte" or "torch" or "file"
        """
        Compress is the ZipNN function used for compression after the configuration is set.

        Parameters
        -------------------------------------
        data: string
                The data to compress. It’s type can be one of the following options: ‘byte’, ‘torch’, ‘file’. If file, enter filename.
                Default is None.

        delta_second_data: string
                If compression is delta compression,then second data is needed. It's type options are ‘byte’, ‘torch’, ‘file’.
                If file, enter filename.
                Default is None.

        compress_cpu_gpu: string
                Compression will be done by choice, in the CPU or GPU.
                Default is cpu.

        lossy_compressed_type: string
                Lossy compression data type, options are ‘byte’, ‘torch’, ‘file’.
                Default is None.

        lossy_compressed_factor: int
                Lossy compression factor.
                Default is None.

        Returns
        -------------------------------------
        Returns the output of one of the following: compress_delta, compress_bin, compress_torch, compress_file
        (depends on the type of the data compressed), which will be the compressed file,
        in the format chosen in the ZipNN class instance configuration.
        """

        if self.delta_compressed_type is not None:
            return self.compress_delta(data, delta_second_data, lossy_compressed_type, lossy_compressed_factor)
        if self.input_format == "byte":
            return self.compress_bin(data)
        elif self.input_format == "torch":
            return self.compress_torch(data, lossy_compressed_type, lossy_compressed_factor)
        elif self.input_format == "file":
            return self.compress_file(self.use_var(data, self.input_file))
        raise ValueError("Unsupported input type")

    def compress_method(self, data: bytes):
        """
        Chooses compression based on compression method.

        Parameters
        -------------------------------------
        data: byte
                Data to compress.

        Returns
        -------------------------------------
        Compression of the data in the chosen method.
        """
        if self.method == EnumMethod.ZSTD.value:
            return self._zstd_compress.compress(data)

        elif self.method == EnumMethod.LZ4.value:
            return lz4.frame.compress(data)

        elif self.method == EnumMethod.SNAPPY.value:
            return snappy.compress(data)
        raise ValueError("Unsupported compression method")

    def compress_bin(self, ba: bytes):
        """
        Compresses byte data.

        Parameters
        -------------------------------------
        ba: byte
                Byte data to compress.

        Returns
        -------------------------------------
        Returns a byte array of the header, data, and some metadata.
        """
        is_print = 0
        if (self.reorder_signbit != 0):
            if(self.reorder_signbit == 32):
                reorder.reorder_float32_bytearray(ba)
            if(self.reorder_signbit == 16):
                reorder.reorder_bfloat16_bytearray(ba)

        if self.bg <= 1:
            ba_comp = self._header + self.compress_method(ba)
        else:
            bg_ret = []
            bg_len = []
            bg_is_comp = []
            for i in range(4):
                stime = time.time()
                ba_bg = ba[i::4]
                bg_comp = self.compress_method(ba_bg)
                if len(bg_comp) / len(ba_bg) < self.bg_compression_threshold:
                    # Save this byte group compressed
                    bg_is_comp.append((1).to_bytes(1, byteorder="little"))
                    bg_len.append(len(bg_comp).to_bytes(length=8, byteorder="little"))
                    bg_ret.append(bg_comp)
                    if is_print:
                        print(f"We compress this byte: {len(bg_comp)/len(ba_bg)} time {time.time()-stime}")
                else:
                    if is_print:
                        print(f"We don't compress this byte: {len(bg_comp)/len(ba_bg)} time {time.time()-stime}")
                    # Save the byte group not compressed
                    bg_is_comp.append((0).to_bytes(1, byteorder="little"))
                    bg_len.append(len(ba_bg).to_bytes(length=8, byteorder="little"))
                    bg_ret.append(ba_bg)
            if self.input_format == "torch":
                ba_comp = b"".join([self._header] + [shape_bytes] + bg_is_comp + bg_len + bg_ret)
            else:
                ba_comp = b"".join([self._header] + bg_is_comp + bg_len + bg_ret)
        if is_print:
            print(f"len ba-comp {len(ba_comp)}")
            print(f"len ba {len(ba)}")

#        if self.compressed_ret_type == "file":
#            with open(self.compressed_file, "wb") as out_file_handler:
#                out_file_handler.write(ba_comp)
#            return 0
        return ba_comp

    def compress_file(self, filename: str):
        """
        Compresses file.

        Parameters
        -------------------------------------
        filename: string
                File name to compress.

        Returns
        -------------------------------------
        Byte array of compressed data.
        """

        raise ImportError("Not implemented Yet")

    #        if not os.path.exists(filename):
    #         raise FileNotFoundError(f"The file at {filename} was not found.")
    #     with open(filename, "rb") as in_file_handler:
    #         data = in_file_handler.read()
    #     ba = self.compress_bin(data)
    #     return ba
    #
    #     # streaming
    #       if (self.is_streaming): # streaming only for input_format == "file_streaming" and compress_ret_type == "file_streaming"
    #          assert (self.compressed_ret_type == "file")
    #          with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
    #              while chunk := infile.read(CHUNK_SIZE):
    #                  compressed_chunk = compressor.compress(chunk)
    #                  if compressed_chunk:
    #                      outfile.write(compressed_chunk)

    # Write any remaining data in the buffer

    #        return (ba_comp)

    def compress_torch(self, data, lossy_compressed_type=None, lossy_compressed_factor=None):
        """
        Compresses torch.

        Parameters
        -------------------------------------
        data: torch.Tensor
                Torch data to compress.

        lossy_compressed_type: string
                ZipNN attribute lossy_compressed_type.
                Default is None.

        lossy_compressed_factor: int
                ZipNN attribute lossy_compressed_factor.
                Default is None.

        Returns
        -------------------------------------
        Byte array of compressed data.
        """
        self._update_header_dtype(data.dtype)
        lossy_type = self.use_var(lossy_compressed_type, self.lossy_compressed_type)
        lossy_type = EnumLossy.NONE if lossy_type is None else lossy_type
        if lossy_type is not EnumLossy.NONE:
            lossy_factor = self.use_var(lossy_compressed_factor, self.lossy_compressed_factor)
            lossy_compress = self.lossy_compress(data, lossy_type, lossy_factor)

            return self.compress_bin(lossy_compress.numpy().tobytes())
        return self.compress_bin(data.numpy().tobytes())

    def lossy_compress(self, data, lossy_type, lossy_factor):
        """
        Handles lossy compression.

        Parameters
        -------------------------------------
        data:
                Data to compress.

        lossy_type: string
                ZipNN attribute lossy_compressed_type.

        lossy_factor: int
                ZipNN attribute lossy_compressed_factor.

        Returns
        -------------------------------------
        Data after lossy compression.
        """
        is_int = False
        if lossy_type == EnumLossy.INTEGER:
            bit_size, lossy_compressed_dtype = zipnn_get_dtype_bits(data.dtype)
            multiplier = 2**lossy_factor
            max_val = bit_size - 1 - lossy_factor
            data, is_int = zipnn_multiply_if_max_below(data, max_val, multiplier, lossy_compressed_dtype)
            self._update_header_lossy(lossy_type, lossy_factor, is_int)

        elif lossy_type == EnumLossy.UNSIGN:
            raise ValueError('lossy_compressed_type is "unsign" is not implemented yet')
        else:
            raise ValueError("Unsupported lossy_compressed_type")

        return data

    def compress_delta(self, delta_second_data, lossy_compressed_type, lossy_compressed_factor):
        """
        Handles delta compression.

        Parameters
        -------------------------------------
        delta_second_data: string
                Type of second data for the delta compression.

        lossy_type: string
                ZipNN attribute lossy_compressed_type.

        lossy_factor: int
                ZipNN attribute lossy_compressed_factor.

        Returns
        -------------------------------------
        Data after lossy compression.
        """
        raise ImportError("Not implemented Yet")

    #################
    # decompression #
    #################

    def decompress(self, data, decompress_cpu_gpu="cpu"):
        """
        Decompress is the ZipNN function used for decompression.

        Parameters
        -------------------------------------
        data: string
            The data to compress. It’s type can be one of the following options: ‘byte’, ‘torch’, ‘file’.
            If file, enter filename. Default is None.

        decompress_cpu_gpu: string
            Compression will be done by choice, in the CPU or GPU.
            Default is cpu.

        Returns
        -------------------------------------
        Returns the output of decompress_bin or decompress_read_file (depends on the type of the data compressed),
        which will be the compressed file, in the format chosen in the ZipNN class instance configuration.
        """
        return self.decompress_bin(data)

    def decompress_method(self, data):
        """
        Chooses decompression based on decompression method.

        Parameters
        -------------------------------------
        data: byte
                Data to decompress.

        Returns
        -------------------------------------
        Decompression of the data in the chosen method.
        """
        if self.method == EnumMethod.ZSTD.value:
            return self._zstd_decompress.decompress(data)
        elif self.method == EnumMethod.LZ4.value:
            return lz4.frame.decompress(data)
        elif self.method == EnumMethod.SNAPPY.value:
            return snappy.decompress(data)
        raise ValueError("Unsupported compression method")

    def decompress_lossy(self, tensor, original_dtype):
        """
        Handles lossy decompression.

        Parameters
        -------------------------------------
        tensor: torch.Tensor
                The tensor data to decompress.

        original_dtype: string
                Original dtype value of the tensor.

        Returns
        -------------------------------------
        Tensor data after lossy decompression.
        """
        if self._is_int == 0:  # no need to transfer to integer from float
            tensor = tensor.view(original_dtype)
            return tensor
        # transfer from integer to float
        bit_size, int_dtype = zipnn_get_dtype_bits(original_dtype)
        tensor = tensor.view(int_dtype)
        lossy_factor = self.lossy_compressed_factor
        divisor = 2**lossy_factor
        decompress_tensor = zipnn_divide_int(tensor, divisor)
        return decompress_tensor

    def write_bin(self, ba_decom):
        """
        Writes decompressed data to file.

        Parameters
        -------------------------------------
        ba_decom: byte
                The data to write to the file.

        Returns
        -------------------------------------
        0 is succeed
        """
        with open(self.decompressed_file, "wb") as out_file_handler:
            out_file_handler.write(ba_decom)
        return 0

    def _revert_reorder_bits(self, ba): 
        if(self.reorder_signbit == 32):
            reorder.reorder_float32_bytearray(ba)
        if(self.reorder_signbit == 16):
            reorder.reorder_bfloat16_bytearray(ba)

    def decompress_bin(self, ba_compress: bytes):
        """
        Decompresses byte data from either a byte array or a tensor.

        Parameters
        -------------------------------------
        ba_compress: byte
                Byte data to decompress.

        Returns
        -------------------------------------
        Returns a byte array of the decompressed data.
        """
        is_print = 0
        stime = time.time()
        header_length = self._retrieve_header(ba_compress)
        start_is_comp = header_length

        if self.bg <= 1:
            ba_decom = self.decompress_method(bytes(ba_compress[start_is_comp:]))
            if (self.reorder_signbit != 0):
                self._revert_reorder_bits(ba_decom)

            if self.decompressed_ret_type == "byte":
                return ba_decom
            if self.decompressed_ret_type == "file":
                return self.write_bin(ba_decom)
            if self.decompressed_ret_type == "tensor":
                numpy_dtype = np.dtype(self.torch_dtype.numpy_dtype)
                tensor = torch.from_numpy(numpy_array)
                return torch
        else:
            ba_bg = []
            start_len = start_is_comp + 4
            start_ba = [start_len + 32]
            end_ba = []
            for i in range(4):
                btime = time.time()
                mv = memoryview(ba_compress)
                is_comp = int.from_bytes(mv[start_is_comp + i : start_is_comp + i + 1], byteorder="little")
                end_ba.append(int.from_bytes(mv[start_len + i * 8 : start_len + (i + 1) * 8 - 1], byteorder="little") + start_ba[i])
                start_ba.append(end_ba[i])
                if is_comp == 1:
                    ba_bg.append(self.decompress_method(ba_compress[start_ba[i] : end_ba[i]]))
                else:
                    ba_bg.append(mv[start_ba[i] : end_ba[i]])
                if is_print:
                    print(f"the time of this byte is: {time.time()-btime}")

            if is_print:
                print(f"The time of decomp is {time.time()-stime} ")
                stime = time.time()
            arr0 = np.frombuffer(ba_bg[0], dtype=np.uint8)
            arr1 = np.frombuffer(ba_bg[1], dtype=np.uint8)
            arr2 = np.frombuffer(ba_bg[2], dtype=np.uint8)
            arr3 = np.frombuffer(ba_bg[3], dtype=np.uint8)

            new_arr = np.empty(arr0.size + arr1.size + arr2.size + arr3.size, dtype=arr1.dtype)
            new_arr[0::4] = arr0
            new_arr[1::4] = arr1
            new_arr[2::4] = arr2
            new_arr[3::4] = arr3

        if self.lossy_compressed_type or self.decompressed_ret_type == "torch":
            tensor = torch.from_numpy(new_arr).to(torch.uint8)
            if self.lossy_compressed_type:
                tensor = self.decompress_lossy(tensor, self.torch_dtype)
            if self.decompressed_ret_type == "torch":
                tensor = tensor.view(self.torch_dtype)
            return tensor

        else:  # return type: Byte and File
            ba_decom = memoryview(new_arr)
            if (self.reorder_signbit != 0):
                self._revert_reorder_bits(ba_decom)
        if self.decompressed_ret_type == "file":
            return self.write_bin(ba_decom)
        return ba_decom

    def decompress_read_file(self, data):
        """
        Decompresses data from file.

        Parameters
        -------------------------------------
        data: string
                The filename to decompress the data from.

        Returns
        -------------------------------------
        Byte array of the decompressed data.
        """
        filename = self.use_var(data, self.compressed_file)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file at {filename} was not found.")
        with open(filename, "rb") as in_file_handler:
            ba = in_file_handler.read()
        return self.decompress_bin(ba)


#    def decompress_delta(self, base_path, delta_file):
#        return 0

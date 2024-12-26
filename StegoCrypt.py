import random
import numpy as np
import qrcode
from PIL import Image
from numpy import ndarray
from pyzbar.pyzbar import decode


class StegoCrypt:
    """
    The StegoCrypt class is responsible for hiding and extracting text messages
    within images using QR codes, a tangential logistic map, and DNA-based encoding.

    Both 'alpha' and 'threshold' must be provided during class initialization
    and are used throughout the encryption/decryption methods.
    """

    def __init__(self, alpha: float, threshold: float) -> None:
        """
        Initializes the StegoCrypt class with the given alpha and threshold values.

        Parameters
        ----------
        alpha : float
            Parameter used in the tangential logistic map equation.
        threshold : float
            Threshold value used to convert the logistic map output into binary form.
        """
        self.alpha = alpha
        self.threshold = threshold

    def _generate_logistic_keystream(self, logistic_seed: int, length: int) -> list[int]:
        """
        Generates a tangential logistic map of a given length and converts it
        into a binary key stream using the defined threshold.

        The logistic map is computed using 'self.alpha' as a multiplier in the
        tangential equation. Each generated value is then compared against
        'self.threshold' to decide whether it becomes '0' or '1'.

        Parameters
        ----------
        logistic_seed : int
            The seed (initial condition) for the logistic map.
        length : int
            Number of elements to generate in the logistic map.

        Returns
        -------
        key_stream : list of int
            A list of bits (0s and 1s) derived from the logistic map.
        """
        x_value = logistic_seed
        logistic_map = []
        # Generate values of the logistic map
        for _ in range(length):
            # Compute the next value using the tangential logistic map equation
            x_value = self.alpha * np.tan(x_value) * (1 - x_value)
            logistic_map.append(x_value)

        # Convert logistic map values into binary (0/1) based on the threshold
        key_stream = [1 if val > self.threshold else 0 for val in logistic_map]
        return key_stream

    def _xor_bitstream(self, data_bits: str, key_bits: list[int]) -> str:
        """
        Applies XOR between a bitstream (as a string of '0' and '1') and a binary key stream.

        This method can be used for both encryption and decryption, since XOR is its own inverse.

        Parameters
        ----------
        data_bits : str
            A string of '0' and '1' representing the data to be processed.
        key_bits : list of int
            A list of bits (0 or 1) representing the key stream.

        Returns
        -------
        result : str
            The result of the XOR operation, also represented as a string of '0' and '1'.
        """
        return ''.join(str(int(d) ^ k) for d, k in zip(data_bits, key_bits))

    def _generate_qr_code(self, qr_data: str, size: int = 256) -> Image.Image:
        """
        Generates a QR code image for the given string data using the qrcode library.

        Parameters
        ----------
        qr_data : str
            The data to be embedded in the QR code (can be plaintext or encrypted bits).
        size : int
            The width and height of the final QR code image in pixels.

        Returns
        -------
        img : PIL.Image.Image
            A PIL Image object containing the generated QR code.
        """
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img = img.resize((size, size))
        return img

    def _qr_to_bitstream_2d(self, qr_image: Image.Image) -> ndarray:
        """
        Converts a QR code PIL image into a 2D NumPy array of bits (0 or 1).

        Black pixels are interpreted as 0, while white pixels are interpreted as 1.

        Parameters
        ----------
        qr_image : PIL.Image.Image
            The PIL Image object containing the QR code.

        Returns
        -------
        bitstream_2d : np.ndarray
            A 2D NumPy array of bits (0 or 1).
        """
        qr_array = np.array(qr_image)
        bitstream_2d = np.where(qr_array == 0, 0, 1)
        return bitstream_2d

    def _bitstream_2d_to_dna(self, bitstream_2d: ndarray) -> ndarray:
        """
        Converts a 2D bitstream (0/1) into a 2D NumPy array of DNA nucleotides.

        The mapping is as follows:
            '00' -> 'A'
            '01' -> 'T'
            '10' -> 'C'
            '11' -> 'G'

        Parameters
        ----------
        bitstream_2d : np.ndarray
            A 2D NumPy array of bits (0 or 1).

        Returns
        -------
        dna_2d : np.ndarray
            A 2D NumPy array of single-character strings representing the DNA nucleotides.
        """
        dna_map = {'00': 'A', '01': 'T', '10': 'C', '11': 'G'}
        rows, cols = bitstream_2d.shape
        dna_matrix = []

        for row in range(rows):
            dna_row = []
            for col in range(0, cols, 2):
                bits = ''.join(map(str, bitstream_2d[row, col:col + 2]))
                dna_row.append(dna_map[bits])
            dna_matrix.append(dna_row)

        dna_2d = np.array(dna_matrix)
        return dna_2d

    def _dna_to_bitstream_2d(self, dna_2d: ndarray) -> ndarray:
        """
        Converts a 2D NumPy array of DNA nucleotides back into a 2D bitstream.

        Parameters
        ----------
        dna_2d : np.ndarray
            A 2D NumPy array of characters ('A', 'T', 'C', 'G').

        Returns
        -------
        bitstream_2d : np.ndarray
            A 2D NumPy array of bits (dtype=np.uint8), where each nucleotide maps
            to a pair of bits according to the following scheme:
                'A' -> 00
                'T' -> 01
                'C' -> 10
                'G' -> 11
        """
        dna_to_bits_map = {'A': '00', 'T': '01', 'C': '10', 'G': '11'}
        bitstream_rows = []

        for dna_row in dna_2d:
            bit_row = []
            for nucleotide in dna_row:
                bits = dna_to_bits_map[nucleotide]
                bit_row.extend(int(bit) for bit in bits)
            bitstream_rows.append(bit_row)

        bitstream_2d = np.array(bitstream_rows, dtype=np.uint8)
        return bitstream_2d

    def _mersenne_twister_bitstream_2d(self, mersenne_seed: int, size: int = 256) -> ndarray:
        """
        Uses Python's Mersenne Twister (random module) to generate a 2D array of bits (0 or 1).

        Parameters
        ----------
        mersenne_seed : int
            The seed for the pseudo-random number generator.
        size : int
            The height and width of the resulting 2D bit array.

        Returns
        -------
        bitstream_2d : np.ndarray
            A 2D NumPy array of bits (0 or 1).
        """
        random.seed(mersenne_seed)
        bitstream_2d = np.array([
            [random.randint(0, 1) for _ in range(size)] for _ in range(size)
        ], dtype=np.uint8)
        return bitstream_2d

    def _perform_dna_transformation(
            self,
            dna_2d_main: ndarray,
            dna_2d_key: ndarray,
            encode: bool = True
    ) -> ndarray:
        """
        Performs either an encoding or decoding operation on a DNA sequence
        using another DNA sequence (generated by Mersenne Twister) as the key.

        When encode=True, the 'encode_mapper' is used, which encrypts the data.
        When encode=False, the 'decode_mapper' is used, which decrypts the data.

        Parameters
        ----------
        dna_2d_main : np.ndarray
            A 2D NumPy array of nucleotides ('A', 'T', 'C', 'G').
        dna_2d_key : np.ndarray
            A 2D NumPy array of nucleotides ('A', 'T', 'C', 'G'), serving as the key.
        encode : bool
            If True, encryption is performed. If False, decryption is performed.

        Returns
        -------
        dna_2d_result : np.ndarray
            A 2D NumPy array of nucleotides ('A', 'T', 'C', 'G'), representing
            the transformed DNA sequence.
        """
        encode_mapper = {
            "A": {"A": "A", "T": "T", "C": "C", "G": "G"},
            "T": {"A": "T", "T": "C", "C": "G", "G": "A"},
            "C": {"A": "C", "T": "G", "C": "A", "G": "T"},
            "G": {"A": "G", "T": "A", "C": "T", "G": "C"},
        }

        decode_mapper = {
            'A': {'A': 'A', 'T': 'G', 'C': 'C', 'G': 'T'},
            'T': {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'},
            'C': {'A': 'C', 'T': 'T', 'C': 'A', 'G': 'G'},
            'G': {'A': 'G', 'T': 'C', 'C': 'T', 'G': 'A'}
        }

        mapper = encode_mapper if encode else decode_mapper

        vectorized_map = np.vectorize(lambda x, y: mapper[x][y])
        dna_2d_result = vectorized_map(dna_2d_main, dna_2d_key)
        return dna_2d_result

    def _image_from_array(self, bit_array_2d: ndarray, output_file: str | None = None) -> Image.Image:
        """
        Converts a 2D NumPy array (containing 0s and 1s) into a grayscale PIL Image ('L' mode).
        Optionally, the image can be saved to an output file.

        Parameters
        ----------
        bit_array_2d : np.ndarray
            A 2D NumPy array of bits (0 or 1).
        output_file : str or None
            If provided, the image will be saved to the specified path. Otherwise, it is not saved.

        Returns
        -------
        img : PIL.Image.Image
            The resulting grayscale image.
        """
        bit_array_2d = np.array(bit_array_2d, dtype=np.uint8)
        image_array = bit_array_2d * 255
        img = Image.fromarray(image_array, mode='L')
        if output_file:
            img.save(output_file)
        return img

    def _decode_qr_from_array(self, qr_bits_2d: ndarray) -> str | None:
        """
        Decodes data from a 2D bit array representing a QR code in grayscale.

        Parameters
        ----------
        qr_bits_2d : np.ndarray
            A 2D NumPy array of bits (0 or 1), typically of size 256x256.

        Returns
        -------
        decoded_data : str or None
            The decoded string from the QR code, or None if decoding fails.
        """
        qr_image = Image.fromarray(np.array(qr_bits_2d, dtype=np.uint8) * 255, mode='L')
        decoded_objects = decode(qr_image)
        if decoded_objects:
            return decoded_objects[0].data.decode("utf-8")
        return None

    def _embed_qr_in_cover_image(
            self,
            qr_code_arrays: list[ndarray],
            cover_image_file: str,
            stego_image_file: str
    ) -> Image.Image:
        """
        Embeds multiple 2D QR code bit arrays into the least significant bits (LSB)
        of each color channel (R, G, B) of the cover image.

        Each 256x256 QR code is expanded to 512x512 and inserted into the LSB of
        the corresponding channel. The resulting image is saved to 'stego_image_file'.

        Parameters
        ----------
        qr_code_arrays : list of np.ndarray
            A list of 2D bit arrays representing the QR codes to be embedded.
        cover_image_file : str
            Path to the cover image file (in RGB format).
        stego_image_file : str
            Path where the modified stego image will be saved.

        Returns
        -------
        stego_image : PIL.Image.Image
            The resulting stego image with the embedded QR codes.
        """
        rgb_image = Image.open(cover_image_file)
        red_channel, green_channel, blue_channel = rgb_image.split()
        channels = [red_channel, green_channel, blue_channel]

        modified_channels = []
        for idx, qr_bits in enumerate(qr_code_arrays):
            # Expand the 256x256 bits to 512x512 by duplicating each bit
            expanded_qr = np.zeros((512, 512), dtype=np.uint8)
            expanded_qr[::2, ::2] = qr_bits
            expanded_qr[1::2, ::2] = qr_bits
            expanded_qr[::2, 1::2] = qr_bits
            expanded_qr[1::2, 1::2] = qr_bits

            channel_array = np.array(channels[idx], dtype=np.uint8)
            # Place the expanded QR bits into the LSB of the current channel
            modified_channel = (channel_array & 254) | expanded_qr
            modified_channels.append(Image.fromarray(modified_channel, mode='L'))

        stego_image = Image.merge("RGB", modified_channels)
        stego_image.save(stego_image_file)
        return stego_image

    def _extract_lsb_bits_from_image(self, stego_image_file: str) -> list[ndarray]:
        """
        Extracts the hidden QR code bits (LSB) from each of the three channels (R, G, B)
        of a stego image. The extracted bits are returned in their original size of 256x256.

        Parameters
        ----------
        stego_image_file : str
            Path to the stego image file.

        Returns
        -------
        extracted_qr_bits : list of np.ndarray
            A list containing three 2D bit arrays (256x256 each), extracted from R, G, and B channels.
        """
        rgb_image = Image.open(stego_image_file)
        red_channel, green_channel, blue_channel = rgb_image.split()
        channels = [red_channel, green_channel, blue_channel]

        extracted_qr_bits = []
        for channel in channels:
            channel_array = np.array(channel, dtype=np.uint8)
            lsb_data = channel_array & 1
            qr_code_bits = lsb_data[::2, ::2]
            qr_code_bits = np.array(qr_code_bits, dtype=np.uint8)
            extracted_qr_bits.append(qr_code_bits)
        return extracted_qr_bits

    def hide_data_in_stego_image(
            self,
            plaintext: str,
            cover_image_file: str,
            stego_image_file: str
    ) -> tuple[list[int], list[int]]:
        """
        Splits the given plaintext into three segments, encrypts each segment,
        and embeds them as QR codes into the RGB channels of the specified cover image.

        Parameters
        ----------
        plaintext : str
            The text to be hidden in the cover image.
        cover_image_file : str
            Path to the original cover image.
        stego_image_file : str
            Path where the resulting stego image will be saved.

        Returns
        -------
        logistic_seeds : list of int
            Seeds used for generating the logistic maps (needed for decryption).
        mersenne_seeds : list of int
            Seeds used by the Mersenne Twister generator (needed for decryption).
        """
        bitstream = ''.join(format(byte, '08b') for byte in plaintext.encode('utf-8'))

        length = len(bitstream)
        segment_size = length // 3
        segment_1 = bitstream[:segment_size]
        segment_2 = bitstream[segment_size:2 * segment_size]
        segment_3 = bitstream[2 * segment_size:]
        segments = [segment_1, segment_2, segment_3]

        logistic_seeds = []
        mersenne_seeds = []
        qr_code_bit_arrays = []

        for idx, segment in enumerate(segments):
            logistic_seed = random.randrange(0, 100000)
            logistic_seeds.append(logistic_seed)

            key_stream = self._generate_logistic_keystream(logistic_seed, len(segment))

            encrypted_segment = self._xor_bitstream(segment, key_stream)

            qr_image = self._generate_qr_code(encrypted_segment)
            self._image_from_array(np.array(qr_image), f"tmp\\result_qr_code_{idx}.png")

            qr_bitstream_2d = self._qr_to_bitstream_2d(qr_image)
            dna_main = self._bitstream_2d_to_dna(qr_bitstream_2d)

            mersenne_seed = random.randrange(0, 100000)
            mersenne_seeds.append(mersenne_seed)

            mersenne_bits = self._mersenne_twister_bitstream_2d(mersenne_seed)
            dna_key = self._bitstream_2d_to_dna(mersenne_bits)

            dna_encoded = self._perform_dna_transformation(dna_main, dna_key, encode=True)
            qr_encoded_2d = self._dna_to_bitstream_2d(dna_encoded)
            qr_code_bit_arrays.append(qr_encoded_2d)

            self._image_from_array(qr_encoded_2d, f"tmp\\encoded_qr_code_{idx}.png")

        self._embed_qr_in_cover_image(qr_code_bit_arrays, cover_image_file, stego_image_file)
        return logistic_seeds, mersenne_seeds

    def extract_message_from_stego_image(
            self,
            stego_image_file: str,
            mersenne_seeds: list[int],
            logistic_seeds: list[int]
    ) -> str:
        """
        Extracts and decrypts the hidden message from the stego image using
        the provided Mersenne Twister and logistic map seeds.

        For each of the three channels (R, G, B), it recovers the hidden
        DNA-encoded QR code, decodes it, and then reconstructs the original
        plaintext segment.

        Parameters
        ----------
        stego_image_file : str
            Path to the stego image containing the hidden data.
        mersenne_seeds : list of int
            Seeds for the Mersenne Twister used during encryption.
        logistic_seeds : list of int
            Seeds for the logistic maps used during encryption.

        Returns
        -------
        decoded_text : str
            The decrypted and reassembled plaintext message.
        """
        decoded_segments = []
        extracted_qr_codes = self._extract_lsb_bits_from_image(stego_image_file)

        for idx, qr_bits in enumerate(extracted_qr_codes):
            dna_encoded = self._bitstream_2d_to_dna(qr_bits)

            mersenne_bits = self._mersenne_twister_bitstream_2d(mersenne_seeds[idx])
            dna_key = self._bitstream_2d_to_dna(mersenne_bits)

            dna_decoded = self._perform_dna_transformation(dna_encoded, dna_key, encode=False)
            decoded_qr_bits = self._dna_to_bitstream_2d(dna_decoded)

            self._image_from_array(decoded_qr_bits, f"tmp\\decoded_qr_code_{idx}.png")

            encrypted_segment = self._decode_qr_from_array(decoded_qr_bits)
            if not encrypted_segment:
                continue

            key_stream = self._generate_logistic_keystream(logistic_seeds[idx], len(encrypted_segment))
            decrypted_segment = self._xor_bitstream(encrypted_segment, key_stream)
            decoded_segments.append(decrypted_segment)

        combined_bit_stream = ''.join(decoded_segments)

        byte_array = bytearray()
        for i in range(0, len(combined_bit_stream), 8):
            byte_chunk = combined_bit_stream[i:i + 8]
            byte_array.append(int(byte_chunk, 2))

        decoded_text = byte_array.decode('utf-8', errors='replace')
        return decoded_text

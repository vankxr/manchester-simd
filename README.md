# manchester-simd
SIMD accelerated Manchester encoder/decoder.

First hands-on experience with x86 SIMD instructions.

### Compilling and running the benchmark

    gcc main.c -O3 -mssse3 -o manchester
    ./manchester

Result (on my machine):

    x                       - [EC 89 53 62 B9 5F 8C 68 29 34 ]
    x_man                   - [56 5A 6A 69 99 A5 96 A6 65 69 99 55 6A 5A 96 6A A6 69 A5 9A ]
    Soft Manchester encode: 2364.924 ms (1816.112 Mbps)
    x_man_ssse3             - [56 5A 6A 69 99 A5 96 A6 65 69 99 55 6A 5A 96 6A A6 69 A5 9A ]
    SSSE3 Manchester encode: 207.630 ms (20685.678 Mbps)
    x_man_s                 - [AC B4 D4 D3 33 4B 2D 4C CA D3 32 AA D4 B5 2C D5 4C D3 4B 34 ]
    x_dec (1)               - [EC 89 53 62 B9 5F 8C 68 29 34 ]
    Soft Manchester decode (non-shifted): 14842.458 ms (289.370 Mbps)
    x_dec_ssse3 (1)         - [EC 89 53 62 B9 5F 8C 68 29 34 ]
    SSSE3 Manchester decode (non-shifted): 1766.913 ms (2430.775 Mbps)
    x_dec_s (0)             - [6C 89 53 62 B9 5F 8C 68 29 34 ]
    Soft Manchester decode (shifted): 14594.166 ms (294.293 Mbps)
    x_dec_s_ssse3 (0)       - [6C 89 53 62 B9 5F 8C 68 29 34 ]
    SSSE3 Manchester decode (shifted): 1764.904 ms (2433.542 Mbps)

Note: `x_dec_s` and `x_dec_s_ssse3`'s first byte does not match `x`, but this result is expected, since the first bit is 1 and the message got shifted 1 bit to the left, the first bit is lost. Although the decoder is able to figure out that the bitstream is not aligned and correct it, it is not capable of recovering the "lost" bit.

### Functions
 - `shift_left` - Shifts the entire buffer 1 bit to the left
 - `shift_left_ssse3` - Shifts the entire buffer 1 bit to the left (SIMD accelerated)
 - `shift_right` - Shifts the entire buffer 1 bit to the right
 - `shift_right_ssse3` - Shifts the entire buffer 1 bit to the right (SIMD accelerated)
 - `manchester_encode` - Manchester encode a buffer of data, requires an output buffer double the size of the input
 - `manchester_encode_ssse3` - Manchester encode a buffer of data, requires an output buffer double the size of the input (SIMD accelerated)
 - `manchester_encode_def` - Manchester encode a buffer of data, requires an output buffer double the size of the input ("Pseudocode" implementation, easier to understand)
 - `manchester_weight` - Correlation between Manchester encoded symbols, determines if bitstream is aligned or not
 - `manchester_weight_ssse3` - Correlation between Manchester encoded symbols, determines if bitstream is aligned or not (SIMD accelerated)
 - `manchester_decode` - Manchester decode a buffer of data, requires an output buffer half the size of the input
 - `manchester_decode_ssse3` - Manchester decode a buffer of data, requires an output buffer half the size of the input (SIMD accelerated)



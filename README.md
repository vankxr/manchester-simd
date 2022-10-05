# manchester-simd
SIMD accelerated Manchester encoder/decoder.

First hands-on experience with x86 SIMD instructions.

### Compilling and running the benchmark

    gcc main.c -O3 -mssse3 -o manchester
    ./manchester

Result (on my machine):

    x                       - [E7 F2 B9 07 3C 9B DF DA 0F 49 ]
    x_man                   - [56 95 55 A6 65 69 AA 95 A5 5A 69 65 59 55 59 66 AA 55 9A 69 ]
    Soft Manchester encode: 2349.864 ms (1827.751 Mbps)
    x_man_ssse3             - [56 95 55 A6 65 69 AA 95 A5 5A 69 65 59 55 59 66 AA 55 9A 69 ]
    SSSE3 Manchester encode: 512.425 ms (8381.651 Mbps)
    x_man_s                 - [AD 2A AB 4C CA D3 55 2B 4A B4 D2 CA B2 AA B2 CD 54 AB 34 D2 ]
    x_dec (1)               - [E7 F2 B9 07 3C 9B DF DA 0F 49 ]
    Soft Manchester decode (non-shifted): 14802.151 ms (290.158 Mbps)
    x_dec_ssse3 (1)         - [E7 F2 B9 07 3C 9B DF DA 0F 49 ]
    SSSE3 Manchester decode (non-shifted): 1754.843 ms (2447.494 Mbps)
    x_dec_s (0)             - [67 F2 B9 07 3C 9B DF DA 0F 49 ]
    Soft Manchester decode (shifted): 14534.648 ms (295.499 Mbps)
    x_dec_s_ssse3 (0)       - [67 F2 B9 07 3C 9B DF DA 0F 49 ]
    SSSE3 Manchester decode (shifted): 1753.491 ms (2449.381 Mbps)

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



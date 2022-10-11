# manchester-simd
SIMD accelerated Manchester encoder/decoder.

First hands-on experience with x86 SIMD instructions.

### Compilling and running the benchmark

    gcc main.c -O3 -mssse3 -o manchester
    ./manchester

Result (on my machine):

    x                       - [BE 87 FA 13 2C 56 2D 7E 86 1D ... 27 90 B3 2F DB BD 14 3F 6E CC ]
    x_man                   - [65 56 6A 95 55 66 A9 A5 A6 5A ... AA 5A 9A 99 A6 5A 59 A6 95 96 ]
    Soft Manchester encode: 2350.273 ms (1827.433 Mbps)
    x_man_ssse3             - [65 56 6A 95 55 66 A9 A5 A6 5A ... AA 5A 9A 99 A6 5A 59 A6 95 96 ]
    SSSE3 Manchester encode: 210.345 ms (20418.680 Mbps)
    Manchester encode - SSSE3 vs Soft: PASSED
    x_man_s                 - [CA AC D5 2A AA CD 53 4B 4C B5 ... 54 B5 35 33 4C B4 B3 4D 2B 2D ]
    x_dec (1)               - [BE 87 FA 13 2C 56 2D 7E 86 1D ... 27 90 B3 2F DB BD 14 3F 6E CC ]
    Soft Manchester decode (non-shifted): 13650.048 ms (314.649 Mbps)
    x_dec_ssse3 (1)         - [BE 87 FA 13 2C 56 2D 7E 86 1D ... 27 90 B3 2F DB BD 14 3F 6E CC ]
    SSSE3 Manchester decode (non-shifted): 1609.608 ms (2668.331 Mbps)
    Manchester decode (non-shifted) - SSSE3 vs Soft: PASSED
    x_dec_s (0)             - [3E 87 FA 13 2C 56 2D 7E 86 1D ... 27 90 B3 2F DB BD 14 3F 6E CC ]
    Soft Manchester decode (shifted): 13443.886 ms (319.474 Mbps)
    x_dec_s_ssse3 (0)       - [3E 87 FA 13 2C 56 2D 7E 86 1D ... 27 90 B3 2F DB BD 14 3F 6E CC ]
    SSSE3 Manchester decode (shifted): 1614.709 ms (2659.902 Mbps)
    Manchester decode (shifted) - SSSE3 vs Soft: PASSED

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



# manchester-simd
SIMD accelerated Manchester encoder/decoder.

First hands-on experience with x86 SIMD instructions.

### Compilling and running the benchmark

    gcc main.c -O3 -mssse3 -o manchester
    ./manchester

Result (on my machine):

    x                       - [F1 D3 77 7B 0F DA 00 E2 00 01 ]
    x_man                   - [55 A9 59 A5 95 95 95 65 AA 55 59 66 AA AA 56 A6 AA AA AA A9 ]
    Soft Manchester encode: 2372.445 ms (1810.355 Mbps)
    x_man_ssse3             - [55 A9 59 A5 95 95 95 65 AA 55 59 66 AA AA 56 A6 AA AA AA A9 ]
    SSSE3 Manchester encode: 206.698 ms (20778.949 Mbps)
    x_man_s                 - [AB 52 B3 4B 2B 2B 2A CB 54 AA B2 CD 55 54 AD 4D 55 55 55 53 ]
    x_dec (1)               - [F1 D3 77 7B 0F DA 00 E2 00 01 ]
    Soft Manchester decode (non-shifted): 15019.063 ms (285.968 Mbps)
    x_dec_ssse3 (1)         - [F1 D3 77 7B 0F DA 00 E2 00 01 ]
    SSSE3 Manchester decode (non-shifted): 1787.019 ms (2403.426 Mbps)
    x_dec_s (0)             - [71 D3 77 7B 0F DA 00 E2 00 01 ]
    Soft Manchester decode (shifted): 14659.830 ms (292.975 Mbps)
    x_dec_s_ssse3 (0)       - [71 D3 77 7B 0F DA 00 E2 00 01 ]
    SSSE3 Manchester decode (shifted): 1802.683 ms (2382.542 Mbps)

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



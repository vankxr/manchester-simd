# manchester-simd
SIMD accelerated Manchester encoder/decoder.

First hands-on experience with x86 SIMD instructions.

### Compilling and running the benchmark

    gcc main.c -O3 -mavx2 -o manchester
    ./manchester

Result (on my machine):

    x                       - [AA C8 14 65 1A B7 8C 0F C5 47 ... 38 9A ED 4C 88 AF 24 08 B3 67 ]
    x_man                   - [66 66 5A 6A A9 9A 96 99 A9 66 ... 95 6A A6 A9 AA 9A 6A A9 65 9A ]
    Soft Manchester encode: 515.136 ms (4168.770 Mbps)
    x_man_def               - [66 66 5A 6A A9 9A 96 99 A9 66 ... 95 6A A6 A9 AA 9A 6A A9 65 9A ]
    Def Manchester encode: 3478.795 ms (617.307 Mbps)
    x_man_sse               - [66 66 5A 6A A9 9A 96 99 A9 66 ... 95 6A A6 A9 AA 9A 6A A9 65 9A ]
    SSSE3 Manchester encode: 57.468 ms (37368.339 Mbps)
    Manchester encode - SSSE3 vs Soft: PASSED
    x_man_avx               - [66 66 5A 6A A9 9A 96 99 A9 66 ... 95 6A A6 A9 AA 9A 6A A9 65 9A ]
    AVX2 Manchester encode: 49.645 ms (43256.797 Mbps)
    Manchester encode - AVX2 vs Soft: PASSED
    x_dec (1)               - [AA C8 14 65 1A B7 8C 0F C5 47 ... 38 9A ED 4C 88 AF 24 08 B3 67 ]
    Soft Manchester decode (non-shifted): 3654.676 ms (587.599 Mbps)
    Manchester decode (non-shifted): PASSED
    x_dec_sse (1)           - [AA C8 14 65 1A B7 8C 0F C5 47 ... 38 9A ED 4C 88 AF 24 08 B3 67 ]
    SSSE3 Manchester decode (non-shifted): 417.980 ms (5137.767 Mbps)
    Manchester decode (non-shifted) - SSSE3 vs Soft: PASSED
    x_dec_avx (1)           - [AA C8 14 65 1A B7 8C 0F C5 47 ... 38 9A ED 4C 88 AF 24 08 B3 67 ]
    AVX2 Manchester decode (non-shifted): 392.434 ms (5472.216 Mbps)
    Manchester decode (non-shifted) - AVX2 vs Soft: PASSED
    x_man_s                 - [CC CC B4 D5 53 35 2D 33 52 CC ... 2A D5 4D 53 55 34 D5 52 CB 35 ]
    x_dec_s (0)             - [2A C8 14 65 1A B7 8C 0F C5 47 ... 38 9A ED 4C 88 AF 24 08 B3 67 ]
    Soft Manchester decode (shifted): 3653.481 ms (587.791 Mbps)
    Manchester decode (shifted): PASSED
    x_dec_s_sse (0)         - [2A C8 14 65 1A B7 8C 0F C5 47 ... 38 9A ED 4C 88 AF 24 08 B3 67 ]
    SSSE3 Manchester decode (shifted): 417.933 ms (5138.344 Mbps)
    Manchester decode (shifted) - SSSE3 vs Soft: PASSED
    x_dec_s_avx (0)         - [2A C8 14 65 1A B7 8C 0F C5 47 ... 38 9A ED 4C 88 AF 24 08 B3 67 ]
    AVX2 Manchester decode (shifted): 455.109 ms (4718.614 Mbps)
    Manchester decode (shifted) - AVX2 vs Soft: PASSED

Note: `x_dec_s`, `x_dec_s_sse` and `x_dec_s_avx`'s first byte does not match `x`, but this result is expected. Since the first bit is 1 and the message got shifted 1 bit to the left, the first bit is lost. Although the decoder is able to figure out that the bitstream is not aligned and correct it, it is not capable of recovering the "lost" bit.

### Functions
 - `shift_left` - Shifts the entire buffer 1 bit to the left
 - `shift_left_ssse3` - Shifts the entire buffer 1 bit to the left (SIMD accelerated)
 - `shift_left_avx2` - Shifts the entire buffer 1 bit to the left (SIMD accelerated)
 - `shift_right` - Shifts the entire buffer 1 bit to the right
 - `shift_right_ssse3` - Shifts the entire buffer 1 bit to the right (SIMD accelerated)
 - `shift_right_avx2` - Shifts the entire buffer 1 bit to the right (SIMD accelerated)
 - `manchester_encode` - Manchester encode a buffer of data, requires an output buffer double the size of the input
 - `manchester_encode_ssse3` - Manchester encode a buffer of data, requires an output buffer double the size of the input (SIMD accelerated)
 - `manchester_encode_avx2` - Manchester encode a buffer of data, requires an output buffer double the size of the input (SIMD accelerated)
 - `manchester_encode_def` - Manchester encode a buffer of data, requires an output buffer double the size of the input ("Pseudocode" implementation, easier to understand)
 - `manchester_weight` - Correlation between Manchester encoded symbols, determines if bitstream is aligned or not
 - `manchester_weight_ssse3` - Correlation between Manchester encoded symbols, determines if bitstream is aligned or not (SIMD accelerated)
 - `manchester_weight_avx2` - Correlation between Manchester encoded symbols, determines if bitstream is aligned or not (SIMD accelerated)
 - `manchester_sync` - Applies two `manchester_weight` at the same time, determining if the input buffer is aligned or not and extracting the shifted version at the same time
 - `manchester_sync_ssse3` - Applies two `manchester_weight_ssse3` at the same time, determining if the input buffer is aligned or not and extracting the shifted version at the same time (SIMD accelerated)
 - `manchester_sync_avx2` - Applies two `manchester_weight_avx2` at the same time, determining if the input buffer is aligned or not and extracting the shifted version at the same time (SIMD accelerated)
 - `manchester_decode` - Manchester decode a buffer of data, requires an output buffer half the size of the input
 - `manchester_decode_ssse3` - Manchester decode a buffer of data, requires an output buffer half the size of the input (SIMD accelerated)
 - `manchester_decode_avx2` - Manchester decode a buffer of data, requires an output buffer half the size of the input (SIMD accelerated)

### Notes
As can be seen from the results above, the AVX2 decoder performs slightly worse than the SSSE3 decoder when the input is misaligned. My guess is that the `manchester_sync_avx2` is slightly slower than the SSSE3 version. When using the AVX2 decoder with the SSSE3 sync instead of the AVX2 sync, the AVX2 decoder will always outperform the SSSE3 decoder, even when the input is shifted. This, however, makes the AVX2 decoder perform worse when the bitstream is **not** shifted. This behaviour is consistent between runs in my own machine (Intel 8th gen mobile CPU), I did not test with other CPU generations.

If you are trying to squeeze the maximum amount of performance possible, make some experiments yourself, mixing and matching the different sync/decode methods, and check what works best for you.

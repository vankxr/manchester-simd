#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

void shift_left(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    for(uint32_t i = 0; i < ulInputSize; i++)
        pubOutput[i] = (pubInput[i] << 1) | (i == (ulInputSize - 1) ? 0 : (pubInput[i + 1] >> 7));
}
void shift_left_ssse3(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    const __m128i m_swap = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    const size_t n_step = sizeof(__m128i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;

    while(i < n)
    {
        // Carry
        if(i > 0)
        {
            if(pubInput[i] & 0x80)
                pubOutput[i - 1] |= 0x01;
        }

        // Load 128-bits of the message from memory
        __m128i x = _mm_lddqu_si128((const __m128i *)&pubInput[i]);

        // Swap endianness
        x = _mm_shuffle_epi8(x, m_swap);

        // Shift-left
        __m128i y = _mm_slli_epi64(x, 1);
        x = _mm_slli_si128(x, 8);
        x = _mm_srli_epi64(x, 63);
        x = _mm_or_si128(y, x);

        // Swap endianness
        x = _mm_shuffle_epi8(x, m_swap);

        // Store result
        _mm_storeu_si128((__m128i *)&pubOutput[i], x);

        // Increment index
        i += n_step;
    }

    if(!r)
        return;

    // Remainder
    uint8_t r_d[sizeof(__m128i)];

    memset(r_d, 0, sizeof(r_d));
    memcpy(r_d, &pubInput[i], r);

    if(i > 0)
    {
        if(pubInput[i] & 0x80)
            pubOutput[i - 1] |= 0x01;
    }

    __m128i x = _mm_lddqu_si128((const __m128i *)r_d);

    x = _mm_shuffle_epi8(x, m_swap);

    __m128i y = _mm_slli_epi64(x, 1);
    x = _mm_slli_si128(x, 8);
    x = _mm_srli_epi64(x, 63);
    x = _mm_or_si128(y, x);

    x = _mm_shuffle_epi8(x, m_swap);

    _mm_storeu_si128((__m128i *)r_d, x);

    memcpy(&pubOutput[i], r_d, r);
}
void shift_left_avx2(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    const __m256i m_swap = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    const size_t n_step = sizeof(__m256i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;

    while(i < n)
    {
        // Carry
        if(i > 0)
        {
            if(pubInput[i] & 0x80)
                pubOutput[i - 1] |= 0x01;
        }

        // Load 256-bits of the message from memory
        __m256i x = _mm256_lddqu_si256((const __m256i *)&pubInput[i]);

        // Swap endianness
        x = _mm256_permute4x64_epi64(x, 0x4E);
        x = _mm256_shuffle_epi8(x, m_swap);

        // Shift-left
        __m256i y = _mm256_slli_epi64(x, 1);
        x = _mm256_slli_si256(x, 8);
        x = _mm256_srli_epi64(x, 63);
        x = _mm256_or_si256(y, x);

        // Swap endianness
        x = _mm256_shuffle_epi8(x, m_swap);
        x = _mm256_permute4x64_epi64(x, 0x4E);

        // Store result
        _mm256_storeu_si256((__m256i *)&pubOutput[i], x);

        // 128-bit lane boundary fix
        if(pubInput[i + (n_step >> 1)] & 0x80)
            pubOutput[i + (n_step >> 1) - 1] |= 0x01;

        // Increment index
        i += n_step;
    }

    if(!r)
        return;

    // Remainder
    uint8_t r_d[sizeof(__m256i)];

    memset(r_d, 0, sizeof(r_d));
    memcpy(r_d, &pubInput[i], r);

    if(i > 0)
    {
        if(pubInput[i] & 0x80)
            pubOutput[i - 1] |= 0x01;
    }

    __m256i x = _mm256_lddqu_si256((const __m256i *)r_d);

    x = _mm256_permute4x64_epi64(x, 0x4E);
    x = _mm256_shuffle_epi8(x, m_swap);

    __m256i y = _mm256_slli_epi64(x, 1);
    x = _mm256_slli_si256(x, 8);
    x = _mm256_srli_epi64(x, 63);
    x = _mm256_or_si256(y, x);

    x = _mm256_shuffle_epi8(x, m_swap);
    x = _mm256_permute4x64_epi64(x, 0x4E);

    _mm256_storeu_si256((__m256i *)r_d, x);

    memcpy(&pubOutput[i], r_d, r);

    if(r > (n_step >> 1))
        if(pubInput[i + (n_step >> 1)] & 0x80)
            pubOutput[i + (n_step >> 1) - 1] |= 0x01;
}

void shift_right(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    for(uint32_t i = 0; i < ulInputSize; i++)
        pubOutput[i] = (pubInput[i] >> 1) | ((i == 0 ? 0 : (pubInput[i - 1] & 1)) << 7);
}
void shift_right_ssse3(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    const __m128i m_swap = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    const size_t n_step = sizeof(__m128i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;

    while(i < n)
    {
        // Load 128-bits of the message from memory
        __m128i x = _mm_lddqu_si128((const __m128i *)&pubInput[i]);

        // Swap endianness
        x = _mm_shuffle_epi8(x, m_swap);

        // Shift-right
        __m128i y = _mm_srli_epi64(x, 1);
        x = _mm_srli_si128(x, 8);
        x = _mm_slli_epi64(x, 63);
        x = _mm_or_si128(y, x);

        // Swap endianness
        x = _mm_shuffle_epi8(x, m_swap);

        // Store result
        _mm_storeu_si128((__m128i *)&pubOutput[i], x);

        // Carry
        if(i > 0)
        {
            if(pubInput[i - 1] & 0x01)
                pubOutput[i] |= 0x80;
        }

        // Increment index
        i += n_step;
    }

    if(!r)
        return;

    // Remainder
    uint8_t r_d[sizeof(__m128i)];

    memset(r_d, 0, sizeof(r_d));
    memcpy(r_d, &pubInput[i], r);

    __m128i x = _mm_lddqu_si128((const __m128i *)r_d);

    x = _mm_shuffle_epi8(x, m_swap);

    __m128i y = _mm_srli_epi64(x, 1);
    x = _mm_srli_si128(x, 8);
    x = _mm_slli_epi64(x, 63);
    x = _mm_or_si128(y, x);

    x = _mm_shuffle_epi8(x, m_swap);

    _mm_storeu_si128((__m128i *)r_d, x);

    memcpy(&pubOutput[i], r_d, r);

    if(i > 0)
    {
        if(pubInput[i - 1] & 0x01)
            pubOutput[i] |= 0x80;
    }
}
void shift_right_avx2(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    const __m256i m_swap = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);

    const size_t n_step = sizeof(__m256i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;

    while(i < n)
    {
        // Load 256-bits of the message from memory
        __m256i x = _mm256_lddqu_si256((const __m256i *)&pubInput[i]);

        // Swap endianness
        x = _mm256_permute4x64_epi64(x, 0x4E);
        x = _mm256_shuffle_epi8(x, m_swap);

        // Shift-right
        __m256i y = _mm256_srli_epi64(x, 1);
        x = _mm256_srli_si256(x, 8);
        x = _mm256_slli_epi64(x, 63);
        x = _mm256_or_si256(y, x);

        // Swap endianness
        x = _mm256_shuffle_epi8(x, m_swap);
        x = _mm256_permute4x64_epi64(x, 0x4E);

        // Store result
        _mm256_storeu_si256((__m256i *)&pubOutput[i], x);

        // 128-bit lane boundary fix
        if(pubInput[i + (n_step >> 1) - 1] & 0x01)
            pubOutput[i + (n_step >> 1)] |= 0x80;

        // Carry
        if(i > 0)
        {
            if(pubInput[i - 1] & 0x01)
                pubOutput[i] |= 0x80;
        }

        // Increment index
        i += n_step;
    }

    if(!r)
        return;

    // Remainder
    uint8_t r_d[sizeof(__m256i)];

    memset(r_d, 0, sizeof(r_d));
    memcpy(r_d, &pubInput[i], r);

    __m256i x = _mm256_lddqu_si256((const __m256i *)r_d);

    x = _mm256_permute4x64_epi64(x, 0x4E);
    x = _mm256_shuffle_epi8(x, m_swap);

    __m256i y = _mm256_srli_epi64(x, 1);
    x = _mm256_srli_si256(x, 8);
    x = _mm256_slli_epi64(x, 63);
    x = _mm256_or_si256(y, x);

    x = _mm256_shuffle_epi8(x, m_swap);
    x = _mm256_permute4x64_epi64(x, 0x4E);

    _mm256_storeu_si256((__m256i *)r_d, x);

    memcpy(&pubOutput[i], r_d, r);

    if(i > 0)
    {
        if(pubInput[i - 1] & 0x01)
            pubOutput[i] |= 0x80;
    }

    if(r > (n_step >> 1))
        if(pubInput[i + (n_step >> 1) - 1] & 0x01)
            pubOutput[i + (n_step >> 1)] |= 0x80;
}

void manchester_encode(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    for(uint32_t i = 0; i < ulInputSize; i++)
    {
        uint16_t x = pubInput[i];

        x = (x ^ (x << 4)) & 0x0F0F;
        x = (x ^ (x << 2)) & 0x3333;
        x = (x ^ (x << 1)) & 0x5555;

        uint16_t y = x ^ 0x5555;

        y <<= 1;
        //x <<= 1;

        uint16_t out = y | x;

        pubOutput[2 * i + 0] = (out >> 8) & 0xFF;
        pubOutput[2 * i + 1] = (out >> 0) & 0xFF;
    }
}
void manchester_encode_ssse3(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    const __m128i m_quad = _mm_set1_epi8(0x0F);
    const __m128i m_dual = _mm_set1_epi8(0x33);
    const __m128i m_single = _mm_set1_epi8(0x55);
    const __m128i m_swap = _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);

    const size_t n_step = sizeof(__m128i) / 2;

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;

    while(i < n)
    {
        // Load 64-bits of the message from memory into the LSB of every 16-bit word
        __m128i x = _mm_loadu_si64(&pubInput[i]);

        // Place each byte at the LSB of a 16-bit word
        x = _mm_unpacklo_epi8(x, _mm_setzero_si128());

        // Clever algorithm to bit-interleave the original message with zeros
        // Credits: https://lemire.me/blog/2018/01/09/how-fast-can-you-bit-interleave-32-bit-integers-simd-edition/
        // Modified to work with SSE2 instructions (128-bit) instead of AVX (256-bit)
        x = _mm_xor_si128(x, _mm_slli_epi16(x, 4));
        x = _mm_and_si128(x, m_quad);
        x = _mm_xor_si128(x, _mm_slli_epi16(x, 2));
        x = _mm_and_si128(x, m_dual);
        x = _mm_xor_si128(x, _mm_slli_epi16(x, 1));
        x = _mm_and_si128(x, m_single);

        // Invert the message bits, ignoring the added zero interleaving
        __m128i y = _mm_xor_si128(x, m_single);

        // Shift inverted message one bit to the left to align
        // This produces (~b, b), which is effectively Manchester encoding (0 = 0b10, 1 = 0b01)
        // For inverted Manchester (0 = 0b01, 1 = 0b10), comment the first line and uncomment the second
        y = _mm_slli_epi16(y, 1);
        //x = _mm_slli_epi16(x, 1);

        // Combine both
        __m128i out = _mm_or_si128(y, x);

        // Swap bytes in 16-bit word (Little endian -> Big endian)
        out = _mm_shuffle_epi8(out, m_swap);

        // Store the result in the output buffer
        _mm_storeu_si128((__m128i *)&pubOutput[i << 1], out);

        // Increment index
        i += n_step;
    }

    if(!r)
        return;

    // Remainder
    uint8_t r_d[sizeof(__m128i)];

    memcpy(r_d, &pubInput[i], r);

    __m128i x = _mm_loadu_si64(r_d);

    x = _mm_unpacklo_epi8(x, _mm_setzero_si128());

    x = _mm_xor_si128(x, _mm_slli_epi16(x, 4));
    x = _mm_and_si128(x, m_quad);
    x = _mm_xor_si128(x, _mm_slli_epi16(x, 2));
    x = _mm_and_si128(x, m_dual);
    x = _mm_xor_si128(x, _mm_slli_epi16(x, 1));
    x = _mm_and_si128(x, m_single);

    __m128i y = _mm_xor_si128(x, m_single);

    y = _mm_slli_epi16(y, 1);
    //x = _mm_slli_epi16(x, 1);

    __m128i out = _mm_or_si128(y, x);

    out = _mm_shuffle_epi8(out, m_swap);

    _mm_storeu_si128((__m128i *)r_d, out);

    memcpy(&pubOutput[i << 1], r_d, r << 1);
}
void manchester_encode_avx2(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    const __m256i m_quad = _mm256_set1_epi8(0x0F);
    const __m256i m_dual = _mm256_set1_epi8(0x33);
    const __m256i m_single = _mm256_set1_epi8(0x55);
    const __m256i m_swap = _mm256_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1,
                                           14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);

    const size_t n_step = sizeof(__m256i) / 2;

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;

    while(i < n)
    {
        // Load 128-bits of the message from memory into the LSB of every 16-bit word
        //__m256i x = _mm256_loadu2_m128i((const __m128i *)&pubInput[i], (const __m128i *)&pubInput[i]);
        __m256i x = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)&pubInput[i])); // Workaround for _mm256_loadu2_m128i not defined

        // Place each byte at the LSB of a 16-bit word
        x = _mm256_permute4x64_epi64(x, 0x10);
        x = _mm256_unpacklo_epi8(x, _mm256_setzero_si256());

        // Clever algorithm to bit-interleave the original message with zeros
        // Credits: https://lemire.me/blog/2018/01/09/how-fast-can-you-bit-interleave-32-bit-integers-simd-edition/
        // Modified to work with SSE2 instructions (256-bit) instead of AVX (256-bit)
        x = _mm256_xor_si256(x, _mm256_slli_epi16(x, 4));
        x = _mm256_and_si256(x, m_quad);
        x = _mm256_xor_si256(x, _mm256_slli_epi16(x, 2));
        x = _mm256_and_si256(x, m_dual);
        x = _mm256_xor_si256(x, _mm256_slli_epi16(x, 1));
        x = _mm256_and_si256(x, m_single);

        // Invert the message bits, ignoring the added zero interleaving
        __m256i y = _mm256_xor_si256(x, m_single);

        // Shift inverted message one bit to the left to align
        // This produces (~b, b), which is effectively Manchester encoding (0 = 0b10, 1 = 0b01)
        // For inverted Manchester (0 = 0b01, 1 = 0b10), comment the first line and uncomment the second
        y = _mm256_slli_epi16(y, 1);
        //x = _mm256_slli_epi16(x, 1);

        // Combine both
        __m256i out = _mm256_or_si256(y, x);

        // Swap bytes in 16-bit word (Little endian -> Big endian)
        out = _mm256_shuffle_epi8(out, m_swap);

        // Store the result in the output buffer
        _mm256_storeu_si256((__m256i *)&pubOutput[i << 1], out);

        // Increment index
        i += n_step;
    }

    if(!r)
        return;

    // Remainder
    uint8_t r_d[sizeof(__m256i)];

    memcpy(r_d, &pubInput[i], r);

    __m256i x = _mm256_lddqu_si256((const __m256i *)r_d);

    x = _mm256_permute4x64_epi64(x, 0x10);
    x = _mm256_unpacklo_epi8(x, _mm256_setzero_si256());

    x = _mm256_xor_si256(x, _mm256_slli_epi16(x, 4));
    x = _mm256_and_si256(x, m_quad);
    x = _mm256_xor_si256(x, _mm256_slli_epi16(x, 2));
    x = _mm256_and_si256(x, m_dual);
    x = _mm256_xor_si256(x, _mm256_slli_epi16(x, 1));
    x = _mm256_and_si256(x, m_single);

    __m256i y = _mm256_xor_si256(x, m_single);

    y = _mm256_slli_epi16(y, 1);
    //x = _mm256_slli_epi16(x, 1);

    __m256i out = _mm256_or_si256(y, x);

    out = _mm256_shuffle_epi8(out, m_swap);

    _mm256_storeu_si256((__m256i *)r_d, out);

    memcpy(&pubOutput[i << 1], r_d, r << 1);
}
void manchester_encode_def(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    for(uint32_t i = 0; i < ulInputSize; i++)
    {
        pubOutput[i * 2 + 0] = 0x00;
        pubOutput[i * 2 + 1] = 0x00;

        for(uint8_t j = 8; j > 0; j--)
        {
            if((pubInput[i] >> (j - 1)) & 1)
                pubOutput[i * 2 + (j < 5)] |= 0b01 << (((j - 1) * 2) % 8);
            else
                pubOutput[i * 2 + (j < 5)] |= 0b10 << (((j - 1) * 2) % 8);
        }
    }
}

int64_t manchester_weight(uint8_t *pubInput, uint32_t ulInputSize)
{
    int64_t llWeight = 0;

    for(uint32_t i = 0; i < ulInputSize; i++)
        for(uint8_t j = 8; j > 0; j -= 2)
            if(((pubInput[i] >> (j - 1)) ^ (pubInput[i] >> (j - 2))) & 1)
                llWeight++;
            else
                llWeight--;

    return llWeight;
}
int64_t manchester_weight_ssse3(uint8_t *pubInput, uint32_t ulInputSize)
{
    const __m128i m_corr_lut = _mm_set_epi8(4, 4, 2, 2, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0);
    const __m128i m_mask_l_odds = _mm_set1_epi8(0x0A);

    const size_t n_step = sizeof(__m128i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;
    int64_t llWeight = 0;

    while(i < n)
    {
        // Load 128-bits of the message from memory
        __m128i x = _mm_lddqu_si128((const __m128i *)&pubInput[i]);

        // Shift left by 1 bit to align for XOR
        __m128i y = _mm_slli_epi64(x, 1);
        __m128i y1 = _mm_slli_si128(x, 8);
        y1 = _mm_srli_epi64(y1, 63);
        y = _mm_or_si128(y, y1);

        // XOR all bits
        x = _mm_xor_si128(x, y);

        // Mask odd bits of the lower nibble
        __m128i l = _mm_and_si128(x, m_mask_l_odds);

        // Apply LUT to calculate the weight per nibble (lower)
        __m128i wl = _mm_shuffle_epi8(m_corr_lut, l);

        // Shift right 4 bits and repeat for the upper nibble
        x = _mm_srli_epi16(x, 4);
        __m128i h = _mm_and_si128(x, m_mask_l_odds);
        __m128i wh = _mm_shuffle_epi8(m_corr_lut, h);

        // Add together and add horizontally
        __m128i w = _mm_sad_epu8(_mm_add_epi8(wl, wh), _mm_setzero_si128());
        w = _mm_add_epi32(w, _mm_unpackhi_epi64(w, _mm_setzero_si128()));

        // Accumulate
        llWeight += _mm_cvtsi128_si32(w);

        // Increment index
        i += n_step;
    }

    if(!r)
        return llWeight - ulInputSize * 4; // Subtract bias

    // Remainder
    uint8_t r_d[sizeof(__m128i)];

    memset(r_d, 0, sizeof(r_d));
    memcpy(r_d, &pubInput[i], r);

    __m128i x = _mm_lddqu_si128((const __m128i *)r_d);

    __m128i y = _mm_slli_epi64(x, 1);
    __m128i y1 = _mm_slli_si128(x, 8);
    y1 = _mm_srli_epi64(y1, 63);
    y = _mm_or_si128(y, y1);

    x = _mm_xor_si128(x, y);

    __m128i l = _mm_and_si128(x, m_mask_l_odds);

    __m128i wl = _mm_shuffle_epi8(m_corr_lut, l);

    x = _mm_srli_epi16(x, 4);
    __m128i h = _mm_and_si128(x, m_mask_l_odds);
    __m128i wh = _mm_shuffle_epi8(m_corr_lut, h);

    __m128i w = _mm_sad_epu8(_mm_add_epi8(wl, wh), _mm_setzero_si128());
    w = _mm_add_epi32(w, _mm_unpackhi_epi64(w, _mm_setzero_si128()));

    llWeight += _mm_cvtsi128_si32(w);

    return llWeight - ulInputSize * 4;
}
int64_t manchester_weight_avx2(uint8_t *pubInput, uint32_t ulInputSize)
{
    const __m256i m_corr_lut = _mm256_set_epi8(4, 4, 2, 2, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0,
                                               4, 4, 2, 2, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0);
    const __m256i m_mask_l_odds = _mm256_set1_epi8(0x0A);
    const __m256i m_lane_carry = _mm256_set_epi64x(0, 1, 0, 0);

    const size_t n_step = sizeof(__m256i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;
    int64_t llWeight = 0;

    while(i < n)
    {
        // Load 256-bits of the message from memory
        __m256i x = _mm256_lddqu_si256((const __m256i *)&pubInput[i]);

        // Shift left by 1 bit to align for XOR
        __m256i y = _mm256_slli_epi64(x, 1);
        __m256i y1 = _mm256_slli_si256(x, 8);
        y1 = _mm256_srli_epi64(y1, 63);
        y = _mm256_or_si256(y, y1);

        // 128-bit lane boundary fix
        if(pubInput[i + (n_step >> 1)] & 0x80)
            y = _mm256_or_si256(y, m_lane_carry);

        // XOR all bits
        x = _mm256_xor_si256(x, y);

        // Mask odd bits of the lower nibble
        __m256i l = _mm256_and_si256(x, m_mask_l_odds);

        // Apply LUT to calculate the weight per nibble (lower)
        __m256i wl = _mm256_shuffle_epi8(m_corr_lut, l);

        // Shift right 4 bits and repeat for the upper nibble
        x = _mm256_srli_epi16(x, 4);
        __m256i h = _mm256_and_si256(x, m_mask_l_odds);
        __m256i wh = _mm256_shuffle_epi8(m_corr_lut, h);

        // Add together and add horizontally
        __m256i w = _mm256_sad_epu8(_mm256_add_epi8(wl, wh), _mm256_setzero_si256());
        w = _mm256_add_epi32(_mm256_unpacklo_epi64(w, _mm256_setzero_si256()), _mm256_unpackhi_epi64(w, _mm256_setzero_si256()));
        __m128i w1 = _mm_add_epi32(_mm256_extracti128_si256(w, 0), _mm256_extracti128_si256(w, 1));

        // Accumulate and subtract bias
        llWeight += _mm_cvtsi128_si32(w1);

        // Increment index
        i += n_step;
    }

    if(!r)
        return llWeight - ulInputSize * 4;

    // Remainder
    uint8_t r_d[sizeof(__m256i)];

    memset(r_d, 0, sizeof(r_d));
    memcpy(r_d, &pubInput[i], r);

    __m256i x = _mm256_lddqu_si256((const __m256i *)r_d);

    __m256i y = _mm256_slli_epi64(x, 1);
    __m256i y1 = _mm256_slli_si256(x, 8);
    y1 = _mm256_srli_epi64(y1, 63);
    y = _mm256_or_si256(y, y1);

    if(r > (n_step >> 1))
        if(pubInput[i + (n_step >> 1)] & 0x80)
            y = _mm256_or_si256(y, m_lane_carry);

    x = _mm256_xor_si256(x, y);

    __m256i l = _mm256_and_si256(x, m_mask_l_odds);

    __m256i wl = _mm256_shuffle_epi8(m_corr_lut, l);

    x = _mm256_srli_epi16(x, 4);
    __m256i h = _mm256_and_si256(x, m_mask_l_odds);
    __m256i wh = _mm256_shuffle_epi8(m_corr_lut, h);

    __m256i w = _mm256_sad_epu8(_mm256_add_epi8(wl, wh), _mm256_setzero_si256());
    w = _mm256_add_epi32(_mm256_unpacklo_epi64(w, _mm256_setzero_si256()), _mm256_unpackhi_epi64(w, _mm256_setzero_si256()));
    __m128i w1 = _mm_add_epi32(_mm256_extracti128_si256(w, 0), _mm256_extracti128_si256(w, 1));

    llWeight += _mm_cvtsi128_si32(w1);

    return llWeight - ulInputSize * 4;
}

uint8_t manchester_sync(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    int64_t llWeight = 0;
    int64_t llWeightS = 0;

    for(uint32_t i = 0; i < ulInputSize; i++)
    {
        uint8_t ubData = pubInput[i];
        uint8_t ubDataS1 = (ubData << 1) | (i == (ulInputSize - 1) ? 0 : (pubInput[i + 1] >> 7));
        uint8_t ubDataS2 = ubDataS1 << 1;

        pubOutput[i] = ubDataS1;

        for(uint8_t j = 8; j > 0; j -= 2)
        {
            if(((ubData >> (j - 1)) ^ (ubDataS1 >> (j - 1))) & 1)
                llWeight++;
            else
                llWeight--;

            if(((ubDataS1 >> (j - 1)) ^ (ubDataS2 >> (j - 1))) & 1)
                llWeightS++;
            else
                llWeightS--;
        }
    }

    return llWeight >= llWeightS;
}
uint8_t manchester_sync_ssse3(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    const __m128i m_swap = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m128i m_carry = _mm_set_epi64x(0, 1);
    const __m128i m_corr_lut = _mm_set_epi8(0, 0, 0, 0, 0, 4, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0);
    const __m128i m_mask_l_odds = _mm_set1_epi8(0x0A);

    const size_t n_step = sizeof(__m128i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;
    uint64_t ullWeight = 0;
    uint64_t ullWeightS = 0;

    while(i < n)
    {
        // Load 128-bits of the message from memory
        __m128i x = _mm_lddqu_si128((const __m128i *)&pubInput[i]);

        // Swap endianness
        x = _mm_shuffle_epi8(x, m_swap);

        // Shift left by 1 bit to align for XOR
        __m128i y = _mm_slli_epi64(x, 1);
        __m128i y1 = _mm_slli_si128(x, 8);
        y1 = _mm_srli_epi64(y1, 63);
        y = _mm_or_si128(y, y1);

        // Carry
        if(ulInputSize > i + n_step)
        {
            if(pubInput[i + n_step] & 0x80)
                y = _mm_or_si128(y, m_carry);
        }

        // Extract first shifted version
        //// Swap endianness
        y1 = _mm_shuffle_epi8(y, m_swap);

        //// Store result
        _mm_storeu_si128((__m128i *)&pubOutput[i], y1);

        // Compute weight metric with the original message and the first shifted version
        //// XOR all bits
        x = _mm_xor_si128(x, y);

        //// Mask odd bits of the lower nibble
        __m128i wl = _mm_and_si128(x, m_mask_l_odds);

        //// Apply LUT to calculate the weight per nibble (lower)
        wl = _mm_shuffle_epi8(m_corr_lut, wl);

        //// Shift right 4 bits and repeat for the upper nibble
        x = _mm_srli_epi16(x, 4);
        __m128i wh = _mm_and_si128(x, m_mask_l_odds);
        wh = _mm_shuffle_epi8(m_corr_lut, wh);

        //// Add together and add horizontally
        __m128i w = _mm_sad_epu8(_mm_add_epi8(wl, wh), _mm_setzero_si128());
        w = _mm_add_epi32(w, _mm_unpackhi_epi64(w, _mm_setzero_si128()));

        //// Accumulate
        ullWeight += _mm_cvtsi128_si32(w);

        // Shift left again
        x = _mm_slli_epi64(y, 1);
        y1 = _mm_slli_si128(y, 8);
        y1 = _mm_srli_epi64(y1, 63);
        x = _mm_or_si128(x, y1);

        // Compute weight metric with the first and second shifted versions
        //// XOR all bits
        x = _mm_xor_si128(x, y);

        //// Mask odd bits of the lower nibble
        wl = _mm_and_si128(x, m_mask_l_odds);

        //// Apply LUT to calculate the weight per nibble (lower)
        wl = _mm_shuffle_epi8(m_corr_lut, wl);

        //// Shift right 4 bits and repeat for the upper nibble
        x = _mm_srli_epi16(x, 4);
        wh = _mm_and_si128(x, m_mask_l_odds);
        wh = _mm_shuffle_epi8(m_corr_lut, wh);

        //// Add together and add horizontally
        w = _mm_sad_epu8(_mm_add_epi8(wl, wh), _mm_setzero_si128());
        w = _mm_add_epi32(w, _mm_unpackhi_epi64(w, _mm_setzero_si128()));

        //// Accumulate
        ullWeightS += _mm_cvtsi128_si32(w);

        // Increment index
        i += n_step;
    }

    if(!r)
        return ullWeight >= ullWeightS;

    // Remainder
    uint8_t r_d[sizeof(__m128i)];

    memset(r_d, 0, sizeof(r_d));
    memcpy(r_d, &pubInput[i], r);

    __m128i x = _mm_lddqu_si128((const __m128i *)r_d);

    x = _mm_shuffle_epi8(x, m_swap);

    __m128i y = _mm_slli_epi64(x, 1);
    __m128i y1 = _mm_slli_si128(x, 8);
    y1 = _mm_srli_epi64(y1, 63);
    y = _mm_or_si128(y, y1);

    y1 = _mm_shuffle_epi8(y, m_swap);

    _mm_storeu_si128((__m128i *)r_d, y1);

    memcpy(&pubOutput[i], r_d, r);

    x = _mm_xor_si128(x, y);

    __m128i wl = _mm_and_si128(x, m_mask_l_odds);

    wl = _mm_shuffle_epi8(m_corr_lut, wl);

    x = _mm_srli_epi16(x, 4);
    __m128i wh = _mm_and_si128(x, m_mask_l_odds);
    wh = _mm_shuffle_epi8(m_corr_lut, wh);

    __m128i w = _mm_sad_epu8(_mm_add_epi8(wl, wh), _mm_setzero_si128());
    w = _mm_add_epi32(w, _mm_unpackhi_epi64(w, _mm_setzero_si128()));

    ullWeight += _mm_cvtsi128_si32(w);

    x = _mm_slli_epi64(y, 1);
    y1 = _mm_slli_si128(y, 8);
    y1 = _mm_srli_epi64(y1, 63);
    x = _mm_or_si128(x, y1);

    x = _mm_xor_si128(x, y);

    wl = _mm_and_si128(x, m_mask_l_odds);

    wl = _mm_shuffle_epi8(m_corr_lut, wl);

    x = _mm_srli_epi16(x, 4);
    wh = _mm_and_si128(x, m_mask_l_odds);
    wh = _mm_shuffle_epi8(m_corr_lut, wh);

    w = _mm_sad_epu8(_mm_add_epi8(wl, wh), _mm_setzero_si128());
    w = _mm_add_epi32(w, _mm_unpackhi_epi64(w, _mm_setzero_si128()));

    ullWeightS += _mm_cvtsi128_si32(w);

    return ullWeight >= ullWeightS;
}
uint8_t manchester_sync_avx2(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput)
{
    const __m256i m_swap = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m256i m_carry = _mm256_set_epi64x(0, 0, 0, 1);
    const __m256i m_lane_carry = _mm256_set_epi64x(0, 1, 0, 0);
    const __m256i m_corr_lut = _mm256_set_epi8(0, 0, 0, 0, 0, 4, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0,
                                               0, 0, 0, 0, 0, 4, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0);
    const __m256i m_mask_l_odds = _mm256_set1_epi8(0x0A);

    const size_t n_step = sizeof(__m256i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;
    uint64_t ullWeight = 0;
    uint64_t ullWeightS = 0;

    while(i < n)
    {
        // Load 128-bits of the message from memory
        __m256i x = _mm256_lddqu_si256((const __m256i *)&pubInput[i]);

        // Swap endianness
        x = _mm256_permute4x64_epi64(x, 0x4E);
        x = _mm256_shuffle_epi8(x, m_swap);

        // Shift left by 1 bit to align for XOR
        __m256i y = _mm256_slli_epi64(x, 1);
        __m256i y1 = _mm256_slli_si256(x, 8);
        y1 = _mm256_srli_epi64(y1, 63);
        y = _mm256_or_si256(y, y1);

        // 128-bit lane boundary fix
        if(pubInput[i + (n_step >> 1)] & 0x80)
            y = _mm256_or_si256(y, m_lane_carry);

        // Carry
        if(ulInputSize > i + n_step)
        {
            if(pubInput[i + n_step] & 0x80)
                y = _mm256_or_si256(y, m_carry);
        }

        // Extract first shifted version
        //// Swap endianness
        y1 = _mm256_permute4x64_epi64(y, 0x4E);
        y1 = _mm256_shuffle_epi8(y1, m_swap);

        //// Store result
        _mm256_storeu_si256((__m256i *)&pubOutput[i], y1);

        // Compute weight metric with the original message and the first shifted version
        //// XOR all bits
        x = _mm256_xor_si256(x, y);

        //// Mask odd bits of the lower nibble
        __m256i wl = _mm256_and_si256(x, m_mask_l_odds);

        //// Apply LUT to calculate the weight per nibble (lower)
        wl = _mm256_shuffle_epi8(m_corr_lut, wl);

        //// Shift right 4 bits and repeat for the upper nibble
        x = _mm256_srli_epi16(x, 4);
        __m256i wh = _mm256_and_si256(x, m_mask_l_odds);
        wh = _mm256_shuffle_epi8(m_corr_lut, wh);

        //// Add together and add horizontally
        __m256i w = _mm256_sad_epu8(_mm256_add_epi8(wl, wh), _mm256_setzero_si256());
        w = _mm256_add_epi32(_mm256_unpacklo_epi64(w, _mm256_setzero_si256()), _mm256_unpackhi_epi64(w, _mm256_setzero_si256()));
        __m128i w1 = _mm_add_epi32(_mm256_extracti128_si256(w, 0), _mm256_extracti128_si256(w, 1));

        //// Accumulate
        ullWeight += _mm_cvtsi128_si32(w1);

        // Shift left again
        x = _mm256_slli_epi64(y, 1);
        y1 = _mm256_slli_si256(y, 8);
        y1 = _mm256_srli_epi64(y1, 63);
        x = _mm256_or_si256(x, y1);

        // 128-bit lane boundary fix
        if(pubInput[i + (n_step >> 1)] & 0x40)
            x = _mm256_or_si256(x, m_lane_carry);

        // Compute weight metric with the first and second shifted versions
        //// XOR all bits
        x = _mm256_xor_si256(x, y);

        //// Mask odd bits of the lower nibble
        wl = _mm256_and_si256(x, m_mask_l_odds);

        //// Apply LUT to calculate the weight per nibble (lower)
        wl = _mm256_shuffle_epi8(m_corr_lut, wl);

        //// Shift right 4 bits and repeat for the upper nibble
        x = _mm256_srli_epi16(x, 4);
        wh = _mm256_and_si256(x, m_mask_l_odds);
        wh = _mm256_shuffle_epi8(m_corr_lut, wh);

        //// Add together and add horizontally
        w = _mm256_sad_epu8(_mm256_add_epi8(wl, wh), _mm256_setzero_si256());
        w = _mm256_add_epi32(_mm256_unpacklo_epi64(w, _mm256_setzero_si256()), _mm256_unpackhi_epi64(w, _mm256_setzero_si256()));
        w1 = _mm_add_epi32(_mm256_extracti128_si256(w, 0), _mm256_extracti128_si256(w, 1));

        //// Accumulate
        ullWeightS += _mm_cvtsi128_si32(w1);

        // Increment index
        i += n_step;
    }

    if(!r)
        return ullWeight >= ullWeightS;

    // Remainder
    uint8_t r_d[sizeof(__m256i)];

    memset(r_d, 0, sizeof(r_d));
    memcpy(r_d, &pubInput[i], r);

    __m256i x = _mm256_lddqu_si256((const __m256i *)r_d);

    x = _mm256_permute4x64_epi64(x, 0x4E);
    x = _mm256_shuffle_epi8(x, m_swap);

    __m256i y = _mm256_slli_epi64(x, 1);
    __m256i y1 = _mm256_slli_si256(x, 8);
    y1 = _mm256_srli_epi64(y1, 63);
    y = _mm256_or_si256(y, y1);

    if(r_d[n_step >> 1] & 0x80)
        y = _mm256_or_si256(y, m_lane_carry);

    y1 = _mm256_permute4x64_epi64(y, 0x4E);
    y1 = _mm256_shuffle_epi8(y1, m_swap);

    _mm256_storeu_si256((__m256i *)r_d, y1);

    memcpy(&pubOutput[i], r_d, r);

    x = _mm256_xor_si256(x, y);

    __m256i wl = _mm256_and_si256(x, m_mask_l_odds);

    wl = _mm256_shuffle_epi8(m_corr_lut, wl);

    x = _mm256_srli_epi16(x, 4);
    __m256i wh = _mm256_and_si256(x, m_mask_l_odds);
    wh = _mm256_shuffle_epi8(m_corr_lut, wh);

    __m256i w = _mm256_sad_epu8(_mm256_add_epi8(wl, wh), _mm256_setzero_si256());
    w = _mm256_add_epi32(_mm256_unpacklo_epi64(w, _mm256_setzero_si256()), _mm256_unpackhi_epi64(w, _mm256_setzero_si256()));
    __m128i w1 = _mm_add_epi32(_mm256_extracti128_si256(w, 0), _mm256_extracti128_si256(w, 1));

    ullWeight += _mm_cvtsi128_si32(w1);

    x = _mm256_slli_epi64(y, 1);
    y1 = _mm256_slli_si256(y, 8);
    y1 = _mm256_srli_epi64(y1, 63);
    x = _mm256_or_si256(x, y1);

    if(r_d[n_step >> 1] & 0x40)
        x = _mm256_or_si256(x, m_lane_carry);

    x = _mm256_xor_si256(x, y);

    wl = _mm256_and_si256(x, m_mask_l_odds);

    wl = _mm256_shuffle_epi8(m_corr_lut, wl);

    x = _mm256_srli_epi16(x, 4);
    wh = _mm256_and_si256(x, m_mask_l_odds);
    wh = _mm256_shuffle_epi8(m_corr_lut, wh);

    w = _mm256_sad_epu8(_mm256_add_epi8(wl, wh), _mm256_setzero_si256());
    w = _mm256_add_epi32(_mm256_unpacklo_epi64(w, _mm256_setzero_si256()), _mm256_unpackhi_epi64(w, _mm256_setzero_si256()));
    w1 = _mm_add_epi32(_mm256_extracti128_si256(w, 0), _mm256_extracti128_si256(w, 1));

    ullWeightS += _mm_cvtsi128_si32(w1);

    return ullWeight >= ullWeightS;
}

void manchester_decode(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput, uint8_t *pubAligned)
{
    uint8_t *pubInputShifted = (uint8_t *)malloc(ulInputSize);
    uint8_t *pubCorrect = NULL;

    if(manchester_sync(pubInput, ulInputSize, pubInputShifted))
    {
        if(pubAligned)
            *pubAligned = 1;

        pubCorrect = pubInput;
    }
    else
    {
        if(pubAligned)
            *pubAligned = 0;

        pubCorrect = pubInputShifted;
    }

    for(uint32_t i = 0; i < ulInputSize; i += 2)
    {
        pubOutput[i >> 1] = 0x00;

        for(uint8_t j = 0; j < 8; j += 2)
        {
            pubOutput[i >> 1] >>= 1;
            pubOutput[i >> 1] |= ((pubCorrect[i + 1] >> j) & 1) << 3;
            pubOutput[i >> 1] |= ((pubCorrect[i] >> j) & 1) << 7;
        }
    }

    if(pubCorrect == pubInputShifted)
        pubOutput[(ulInputSize >> 1) - 1] &= 0xFE;

    free(pubInputShifted);
}
void manchester_decode_ssse3(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput, uint8_t *pubAligned)
{
    uint8_t *pubInputShifted = (uint8_t *)malloc(ulInputSize);
    uint8_t *pubCorrect = NULL;

    if(manchester_sync_ssse3(pubInput, ulInputSize, pubInputShifted))
    {
        if(pubAligned)
            *pubAligned = 1;

        pubCorrect = pubInput;
    }
    else
    {
        if(pubAligned)
            *pubAligned = 0;

        pubCorrect = pubInputShifted;
    }

    const __m128i m_quad = _mm_set1_epi8(0x0F);
    const __m128i m_dual = _mm_set1_epi8(0x33);
    const __m128i m_single = _mm_set1_epi8(0x55);
    const __m128i m_swap = _mm_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    const __m128i m_pick = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);

    const size_t n_step = sizeof(__m128i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;

    while(i < n)
    {
        // Load 128-bits of the message from memory
        __m128i x = _mm_lddqu_si128((const __m128i *)&pubCorrect[i]);

        // Swap bytes in 16-bit word (Big endian -> Little endian)
        x = _mm_shuffle_epi8(x, m_swap);

        // Bit-deinterleave
        x = _mm_and_si128(x, m_single);
        x = _mm_or_si128(x, _mm_srli_epi16(x, 1));
        x = _mm_and_si128(x, m_dual);
        x = _mm_or_si128(x, _mm_srli_epi16(x, 2));
        x = _mm_and_si128(x, m_quad);
        x = _mm_or_si128(x, _mm_srli_epi16(x, 4));

        // Pick LSBs from 16-bit words
        __m128i out = _mm_shuffle_epi8(x, m_pick);

        // Store result
        _mm_storeu_si64(&pubOutput[i >> 1], out);

        // Increment index
        i += n_step;
    }

    if(!r)
    {
        if(pubCorrect == pubInputShifted)
            pubOutput[(ulInputSize >> 1) - 1] &= 0xFE;

        free(pubInputShifted);

        return;
    }

    // Remainder
    uint8_t r_d[sizeof(__m128i)];

    memcpy(r_d, &pubCorrect[i], r);

    __m128i x = _mm_lddqu_si128((const __m128i *)r_d);

    x = _mm_shuffle_epi8(x, m_swap);

    x = _mm_and_si128(x, m_single);
    x = _mm_or_si128(x, _mm_srli_epi16(x, 1));
    x = _mm_and_si128(x, m_dual);
    x = _mm_or_si128(x, _mm_srli_epi16(x, 2));
    x = _mm_and_si128(x, m_quad);
    x = _mm_or_si128(x, _mm_srli_epi16(x, 4));

    __m128i out = _mm_shuffle_epi8(x, m_pick);

    _mm_storeu_si64(r_d, out);

    memcpy(&pubOutput[i >> 1], r_d, r >> 1);

    if(pubCorrect == pubInputShifted)
        pubOutput[(ulInputSize >> 1) - 1] &= 0xFE;

    free(pubInputShifted);
}
void manchester_decode_avx2(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput, uint8_t *pubAligned)
{
    uint8_t *pubInputShifted = (uint8_t *)malloc(ulInputSize);
    uint8_t *pubCorrect = NULL;

    if(manchester_sync_avx2(pubInput, ulInputSize, pubInputShifted))
    {
        if(pubAligned)
            *pubAligned = 1;

        pubCorrect = pubInput;
    }
    else
    {
        if(pubAligned)
            *pubAligned = 0;

        pubCorrect = pubInputShifted;
    }

    const __m256i m_quad = _mm256_set1_epi8(0x0F);
    const __m256i m_dual = _mm256_set1_epi8(0x33);
    const __m256i m_single = _mm256_set1_epi8(0x55);
    const __m256i m_swap = _mm256_set_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1,
                                           14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    const __m256i m_pick = _mm256_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0,
                                           0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 12, 10, 8, 6, 4, 2, 0);

    const size_t n_step = sizeof(__m256i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;

    while(i < n)
    {
        // Load 256-bits of the message from memory
        __m256i x = _mm256_lddqu_si256((const __m256i *)&pubCorrect[i]);

        // Swap bytes in 16-bit word (Big endian -> Little endian)
        x = _mm256_shuffle_epi8(x, m_swap);

        // Bit-deinterleave
        x = _mm256_and_si256(x, m_single);
        x = _mm256_or_si256(x, _mm256_srli_epi16(x, 1));
        x = _mm256_and_si256(x, m_dual);
        x = _mm256_or_si256(x, _mm256_srli_epi16(x, 2));
        x = _mm256_and_si256(x, m_quad);
        x = _mm256_or_si256(x, _mm256_srli_epi16(x, 4));

        // Pick LSBs from 16-bit words
        __m256i out = _mm256_shuffle_epi8(x, m_pick);

        // Bring upper 128-bit lower half into lower 128-bits upper half
        out = _mm256_permute4x64_epi64(out, 0xF8);

        // Store result
        _mm_storeu_si128((__m128i *)&pubOutput[i >> 1], _mm256_extracti128_si256(out, 0));

        // Increment index
        i += n_step;
    }

    if(!r)
    {
        if(pubCorrect == pubInputShifted)
            pubOutput[(ulInputSize >> 1) - 1] &= 0xFE;

        free(pubInputShifted);

        return;
    }

    // Remainder
    uint8_t r_d[sizeof(__m256i)];

    memcpy(r_d, &pubCorrect[i], r);

    __m256i x = _mm256_lddqu_si256((const __m256i *)r_d);

    x = _mm256_shuffle_epi8(x, m_swap);

    x = _mm256_and_si256(x, m_single);
    x = _mm256_or_si256(x, _mm256_srli_epi16(x, 1));
    x = _mm256_and_si256(x, m_dual);
    x = _mm256_or_si256(x, _mm256_srli_epi16(x, 2));
    x = _mm256_and_si256(x, m_quad);
    x = _mm256_or_si256(x, _mm256_srli_epi16(x, 4));

    __m256i out = _mm256_shuffle_epi8(x, m_pick);

    out = _mm256_permute4x64_epi64(out, 0xF8);

    _mm256_storeu_si256((__m256i *)r_d, out);

    memcpy(&pubOutput[i >> 1], r_d, r >> 1);

    if(pubCorrect == pubInputShifted)
        pubOutput[(ulInputSize >> 1) - 1] &= 0xFE;

    free(pubInputShifted);
}

void differential_encode(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput, uint8_t *pubPrev)
{
    for(uint32_t i = 0; i < ulInputSize; i++)
    {
        pubOutput[i] = pubInput[i] ^ ((pubInput[i] >> 1) | (*pubPrev << 7));

        *pubPrev = pubInput[i] & 1;
    }
}
void differential_encode_ssse3(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput, uint8_t *pubPrev)
{
    const __m128i m_swap = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m128i m_carry = _mm_set_epi64x(0x8000000000000000, 0);

    const size_t n_step = sizeof(__m128i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;

    while(i < n)
    {
        // Load 128-bits of the message from memory
        __m128i x = _mm_lddqu_si128((const __m128i *)&pubInput[i]);

        // Swap endianness
        x = _mm_shuffle_epi8(x, m_swap);

        // Shift-right
        __m128i y = _mm_srli_epi64(x, 1);
        __m128i y1 = _mm_srli_si128(x, 8);
        y1 = _mm_slli_epi64(y1, 63);
        y = _mm_or_si128(y, y1);

        // Insert carry
        if(*pubPrev)
            y = _mm_or_si128(y, m_carry);

        // XOR (differential encoding)
        x = _mm_xor_si128(x, y);

        // Swap endianness
        x = _mm_shuffle_epi8(x, m_swap);

        // Store result
        _mm_storeu_si128((__m128i *)&pubOutput[i], x);

        // Carry
        *pubPrev = pubInput[i + n_step - 1] & 1;

        // Increment index
        i += n_step;
    }

    if(!r)
        return;

    // Remainder
    uint8_t r_d[sizeof(__m128i)];

    memset(r_d, 0, sizeof(r_d));
    memcpy(r_d, &pubInput[i], r);

    __m128i x = _mm_lddqu_si128((const __m128i *)r_d);

    x = _mm_shuffle_epi8(x, m_swap);

    __m128i y = _mm_srli_epi64(x, 1);
    __m128i y1 = _mm_srli_si128(x, 8);
    y1 = _mm_slli_epi64(y1, 63);
    y = _mm_or_si128(y, y1);

    if(*pubPrev)
        y = _mm_or_si128(y, m_carry);

    x = _mm_xor_si128(x, y);

    x = _mm_shuffle_epi8(x, m_swap);

    _mm_storeu_si128((__m128i *)r_d, x);

    *pubPrev = pubInput[ulInputSize - 1] & 1;

    memcpy(&pubOutput[i], r_d, r);
}
void differential_encode_avx2(uint8_t *pubInput, uint32_t ulInputSize, uint8_t *pubOutput, uint8_t *pubPrev)
{
    const __m256i m_swap = _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    const __m256i m_carry = _mm256_set_epi64x(0x8000000000000000, 0, 0, 0);
    const __m256i m_lane_carry = _mm256_set_epi64x(0, 0, 0x8000000000000000, 0);

    const size_t n_step = sizeof(__m256i);

    uint32_t n = ulInputSize & ~(n_step - 1);
    uint8_t r = ulInputSize & (n_step - 1);
    uint32_t i = 0;

    while(i < n)
    {
        // Load 256-bits of the message from memory
        __m256i x = _mm256_lddqu_si256((const __m256i *)&pubInput[i]);

        // Swap endianness
        x = _mm256_permute4x64_epi64(x, 0x4E);
        x = _mm256_shuffle_epi8(x, m_swap);

        // Shift-right
        __m256i y = _mm256_srli_epi64(x, 1);
        __m256i y1 = _mm256_srli_si256(x, 8);
        y1 = _mm256_slli_epi64(y1, 63);
        y = _mm256_or_si256(y, y1);

        // 128-bit lane boundary fix
        if(pubInput[i + (n_step >> 1) - 1] & 0x01)
            y = _mm256_or_si256(y, m_lane_carry);

        // Insert carry
        if(*pubPrev)
            y = _mm256_or_si256(y, m_carry);

        // XOR (differential encoding)
        x = _mm256_xor_si256(x, y);

        // Swap endianness
        x = _mm256_shuffle_epi8(x, m_swap);
        x = _mm256_permute4x64_epi64(x, 0x4E);

        // Store result
        _mm256_storeu_si256((__m256i *)&pubOutput[i], x);

        // Carry
        *pubPrev = pubInput[i + n_step - 1] & 1;

        // Increment index
        i += n_step;
    }

    if(!r)
        return;

    // Remainder
    uint8_t r_d[sizeof(__m256i)];

    memset(r_d, 0, sizeof(r_d));
    memcpy(r_d, &pubInput[i], r);

    __m256i x = _mm256_lddqu_si256((const __m256i *)r_d);

    x = _mm256_permute4x64_epi64(x, 0x4E);
    x = _mm256_shuffle_epi8(x, m_swap);

    __m256i y = _mm256_srli_epi64(x, 1);
    __m256i y1 = _mm256_srli_si256(x, 8);
    y1 = _mm256_slli_epi64(y1, 63);
    y = _mm256_or_si256(y, y1);

    if(r > (n_step >> 1))
        if(pubInput[i + (n_step >> 1) - 1] & 0x01)
            y = _mm256_or_si256(y, m_lane_carry);

    if(*pubPrev)
        y = _mm256_or_si256(y, m_carry);

    x = _mm256_xor_si256(x, y);

    x = _mm256_shuffle_epi8(x, m_swap);
    x = _mm256_permute4x64_epi64(x, 0x4E);

    _mm256_storeu_si256((__m256i *)r_d, x);

    *pubPrev = pubInput[ulInputSize - 1] & 1;

    memcpy(&pubOutput[i], r_d, r);
}

#define MIN(a,b) (((a)<(b))?(a):(b))

int main(int argc, char *argv[])
{
    // Manchester test
    static const size_t x_sz = 1024 * 1024 * 256 + 7;
    //uint8_t x[10] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    //uint8_t x[10] = {0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0xBC};
    uint8_t *x = malloc(x_sz);
    uint8_t *x_man = malloc(2 * x_sz);
    uint8_t *x_man_def = malloc(2 * x_sz);
    uint8_t *x_man_sse = malloc(2 * x_sz);
    uint8_t *x_man_avx = malloc(2 * x_sz);
    uint8_t *x_man_s = malloc(2 * x_sz);
    uint8_t *x_dec = malloc(x_sz);
    uint8_t *x_dec_sse = malloc(x_sz);
    uint8_t *x_dec_avx = malloc(x_sz);
    uint8_t *x_dec_s = malloc(x_sz);
    uint8_t *x_dec_s_sse = malloc(x_sz);
    uint8_t *x_dec_s_avx = malloc(x_sz);
    uint8_t x_aligned;
    uint8_t pass;
    double elapsed_time;
    clock_t start_time;

    srand((unsigned)time(NULL));

    // Generate data
    for(size_t i = 0; i < x_sz; i++)
        x[i] = rand() % UINT8_MAX;

    printf("x\t\t\t- [");
    for(size_t i = 0; i < MIN(x_sz, 10); i++)
        printf("%02X ", x[i]);
    printf("... ");
    for(size_t i = MIN(x_sz, 10); i > 0; i--)
        printf("%02X ", x[x_sz - i]);
    printf("]\r\n");

    // Soft Manchester benchmark
    memset(x_man, 0, 2*x_sz);
    start_time = clock();
    manchester_encode(x, x_sz, x_man);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    printf("x_man\t\t\t- [");
    for(size_t i = 0; i < MIN(2*x_sz, 10); i++)
        printf("%02X ", x_man[i]);
    printf("... ");
    for(size_t i = MIN(2*x_sz, 10); i > 0; i--)
        printf("%02X ", x_man[x_sz - i]);
    printf("]\r\n");

    printf("Soft Manchester encode: %.3f ms (%.3f Mbps)\r\n", elapsed_time * 1000.0, x_sz * 8 / elapsed_time / 1e6);

    // Definition Manchester benchmark
    memset(x_man_def, 0, 2*x_sz);
    start_time = clock();
    manchester_encode_def(x, x_sz, x_man_def);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    printf("x_man_def\t\t- [");
    for(size_t i = 0; i < MIN(2*x_sz, 10); i++)
        printf("%02X ", x_man_def[i]);
    printf("... ");
    for(size_t i = MIN(2*x_sz, 10); i > 0; i--)
        printf("%02X ", x_man_def[x_sz - i]);
    printf("]\r\n");

    printf("Def Manchester encode: %.3f ms (%.3f Mbps)\r\n", elapsed_time * 1000.0, x_sz * 8 / elapsed_time / 1e6);

    // SSSE3 Manchester benchmark & test
    memset(x_man_sse, 0, 2*x_sz);
    start_time = clock();
    manchester_encode_ssse3(x, x_sz, x_man_sse);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    printf("x_man_sse\t\t- [");
    for(size_t i = 0; i < MIN(2*x_sz, 10); i++)
        printf("%02X ", x_man_sse[i]);
    printf("... ");
    for(size_t i = MIN(2*x_sz, 10); i > 0; i--)
        printf("%02X ", x_man_sse[x_sz - i]);
    printf("]\r\n");

    printf("SSSE3 Manchester encode: %.3f ms (%.3f Mbps)\r\n", elapsed_time * 1000.0, x_sz * 8 / elapsed_time / 1e6);

    pass = 1;
    for(size_t i = 0; i < 2*x_sz; i++)
    {
        if(x_man_sse[i] != x_man[i])
        {
            pass = 0;

            break;
        }
    }

    printf("Manchester encode - SSSE3 vs Soft: %s\r\n", pass ? "PASSED" : "FAILED !!!!!!!!!!");

    // AVX2 Manchester benchmark & test
    memset(x_man_avx, 0, 2*x_sz);
    start_time = clock();
    manchester_encode_avx2(x, x_sz, x_man_avx);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    printf("x_man_avx\t\t- [");
    for(size_t i = 0; i < MIN(2*x_sz, 10); i++)
        printf("%02X ", x_man_avx[i]);
    printf("... ");
    for(size_t i = MIN(2*x_sz, 10); i > 0; i--)
        printf("%02X ", x_man_avx[x_sz - i]);
    printf("]\r\n");

    printf("AVX2 Manchester encode: %.3f ms (%.3f Mbps)\r\n", elapsed_time * 1000.0, x_sz * 8 / elapsed_time / 1e6);

    pass = 1;
    for(size_t i = 0; i < 2*x_sz; i++)
    {
        if(x_man_avx[i] != x_man[i])
        {
            pass = 0;

            break;
        }
    }

    printf("Manchester encode - AVX2 vs Soft: %s\r\n", pass ? "PASSED" : "FAILED !!!!!!!!!!");

    // Soft Manchester decode benchmark & test
    x_aligned = 0;
    memset(x_dec, 0, x_sz);
    start_time = clock();
    manchester_decode(x_man, 2*x_sz, x_dec, &x_aligned);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    printf("x_dec (%hhu)\t\t- [", x_aligned);
    for(size_t i = 0; i < MIN(x_sz, 10); i++)
        printf("%02X ", x_dec[i]);
    printf("... ");
    for(size_t i = MIN(x_sz, 10); i > 0; i--)
        printf("%02X ", x_dec[x_sz - i]);
    printf("]\r\n");

    printf("Soft Manchester decode (non-shifted): %.3f ms (%.3f Mbps)\r\n", elapsed_time * 1000.0, x_sz * 8 / elapsed_time / 1e6);

    pass = 1;
    for(size_t i = 0; i < x_sz; i++)
    {
        if(x_dec[i] != x[i])
        {
            pass = 0;

            break;
        }
    }

    printf("Manchester decode (non-shifted): %s\r\n", pass ? "PASSED" : "FAILED !!!!!!!!!!");

    // SSSE3 Manchester decode benchmark & test
    x_aligned = 0;
    memset(x_dec_sse, 0, x_sz);
    start_time = clock();
    manchester_decode_ssse3(x_man, 2*x_sz, x_dec_sse, &x_aligned);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    printf("x_dec_sse (%hhu)\t\t- [", x_aligned);
    for(size_t i = 0; i < MIN(x_sz, 10); i++)
        printf("%02X ", x_dec_sse[i]);
    printf("... ");
    for(size_t i = MIN(x_sz, 10); i > 0; i--)
        printf("%02X ", x_dec_sse[x_sz - i]);
    printf("]\r\n");

    printf("SSSE3 Manchester decode (non-shifted): %.3f ms (%.3f Mbps)\r\n", elapsed_time * 1000.0, x_sz * 8 / elapsed_time / 1e6);

    pass = 1;
    for(size_t i = 0; i < x_sz; i++)
    {
        if(x_dec_sse[i] != x_dec[i])
        {
            pass = 0;

            break;
        }
    }

    printf("Manchester decode (non-shifted) - SSSE3 vs Soft: %s\r\n", pass ? "PASSED" : "FAILED !!!!!!!!!!");

    // AVX2 Manchester decode benchmark & test
    x_aligned = 0;
    memset(x_dec_avx, 0, x_sz);
    start_time = clock();
    manchester_decode_avx2(x_man, 2*x_sz, x_dec_avx, &x_aligned);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    printf("x_dec_avx (%hhu)\t\t- [", x_aligned);
    for(size_t i = 0; i < MIN(x_sz, 10); i++)
        printf("%02X ", x_dec_avx[i]);
    printf("... ");
    for(size_t i = MIN(x_sz, 10); i > 0; i--)
        printf("%02X ", x_dec_avx[x_sz - i]);
    printf("]\r\n");

    printf("AVX2 Manchester decode (non-shifted): %.3f ms (%.3f Mbps)\r\n", elapsed_time * 1000.0, x_sz * 8 / elapsed_time / 1e6);

    pass = 1;
    for(size_t i = 0; i < x_sz; i++)
    {
        if(x_dec_avx[i] != x_dec[i])
        {
            pass = 0;

            break;
        }
    }

    printf("Manchester decode (non-shifted) - AVX2 vs Soft: %s\r\n", pass ? "PASSED" : "FAILED !!!!!!!!!!");

    // Produce the shifted version of the coded data
    shift_left(x_man, 2*x_sz, x_man_s);

    printf("x_man_s\t\t\t- [");
    for(size_t i = 0; i < MIN(2*x_sz, 10); i++)
        printf("%02X ", x_man_s[i]);
    printf("... ");
    for(size_t i = MIN(2*x_sz, 10); i > 0; i--)
        printf("%02X ", x_man_s[x_sz - i]);
    printf("]\r\n");

    // Soft Manchester decode benchmark & test (shifted)
    x_aligned = 0;
    memset(x_dec, 0, x_sz);
    start_time = clock();
    manchester_decode(x_man_s, 2*x_sz, x_dec, &x_aligned);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    shift_right(x_dec, x_sz, x_dec_s);

    printf("x_dec_s (%hhu)\t\t- [", x_aligned);
    for(size_t i = 0; i < MIN(x_sz, 10); i++)
        printf("%02X ", x_dec_s[i]);
    printf("... ");
    for(size_t i = MIN(x_sz, 10); i > 0; i--)
        printf("%02X ", x_dec_s[x_sz - i]);
    printf("]\r\n");

    printf("Soft Manchester decode (shifted): %.3f ms (%.3f Mbps)\r\n", elapsed_time * 1000.0, x_sz * 8 / elapsed_time / 1e6);

    pass = 1;
    for(size_t i = 0; i < x_sz; i++)
    {
        if(x_dec_s[i] != (x[i] & (!i ? 0x7F : 0xFF)))
        {
            pass = 0;

            break;
        }
    }

    printf("Manchester decode (shifted): %s\r\n", pass ? "PASSED" : "FAILED !!!!!!!!!!");

    // SSSE3 Manchester decode benchmark & test (shifted)
    x_aligned = 0;
    memset(x_dec, 0, x_sz);
    start_time = clock();
    manchester_decode_ssse3(x_man_s, 2*x_sz, x_dec, &x_aligned);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    shift_right(x_dec, x_sz, x_dec_s_sse);

    printf("x_dec_s_sse (%hhu)\t\t- [", x_aligned);
    for(size_t i = 0; i < MIN(x_sz, 10); i++)
        printf("%02X ", x_dec_s_sse[i]);
    printf("... ");
    for(size_t i = MIN(x_sz, 10); i > 0; i--)
        printf("%02X ", x_dec_s_sse[x_sz - i]);
    printf("]\r\n");

    printf("SSSE3 Manchester decode (shifted): %.3f ms (%.3f Mbps)\r\n", elapsed_time * 1000.0, x_sz * 8 / elapsed_time / 1e6);

    pass = 1;
    for(size_t i = 0; i < x_sz; i++)
    {
        if(x_dec_s_sse[i] != x_dec_s[i])
        {
            pass = 0;

            break;
        }
    }

    printf("Manchester decode (shifted) - SSSE3 vs Soft: %s\r\n", pass ? "PASSED" : "FAILED !!!!!!!!!!");

    // AVX2 Manchester decode benchmark & test (shifted)
    x_aligned = 0;
    memset(x_dec, 0, x_sz);
    start_time = clock();
    manchester_decode_avx2(x_man_s, 2*x_sz, x_dec, &x_aligned);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    shift_right(x_dec, x_sz, x_dec_s_avx);

    printf("x_dec_s_avx (%hhu)\t\t- [", x_aligned);
    for(size_t i = 0; i < MIN(x_sz, 10); i++)
        printf("%02X ", x_dec_s_avx[i]);
    printf("... ");
    for(size_t i = MIN(x_sz, 10); i > 0; i--)
        printf("%02X ", x_dec_s_avx[x_sz - i]);
    printf("]\r\n");

    printf("AVX2 Manchester decode (shifted): %.3f ms (%.3f Mbps)\r\n", elapsed_time * 1000.0, x_sz * 8 / elapsed_time / 1e6);

    pass = 1;
    for(size_t i = 0; i < x_sz; i++)
    {
        if(x_dec_s_avx[i] != x_dec_s[i])
        {
            pass = 0;

            break;
        }
    }

    printf("Manchester decode (shifted) - AVX2 vs Soft: %s\r\n", pass ? "PASSED" : "FAILED !!!!!!!!!!");

    return 0;
}
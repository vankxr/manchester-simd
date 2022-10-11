#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>

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

        // Shift left again
        __m128i z = _mm_slli_epi64(y, 1);
        __m128i z1 = _mm_slli_si128(y, 8);
        z1 = _mm_srli_epi64(z1, 63);
        z = _mm_or_si128(z, z1);

        // XOR all bits
        x = _mm_xor_si128(x, y);
        y = _mm_xor_si128(y, z);

        // Mask odd bits of the lower nibble
        __m128i x_l = _mm_and_si128(x, m_mask_l_odds);
        __m128i y_l = _mm_and_si128(y, m_mask_l_odds);

        // Apply LUT to calculate the weight per nibble (lower)
        __m128i x_wl = _mm_shuffle_epi8(m_corr_lut, x_l);
        __m128i y_wl = _mm_shuffle_epi8(m_corr_lut, y_l);

        // Shift right 4 bits and repeat for the upper nibble
        x = _mm_srli_epi16(x, 4);
        __m128i x_h = _mm_and_si128(x, m_mask_l_odds);
        __m128i x_wh = _mm_shuffle_epi8(m_corr_lut, x_h);
        y = _mm_srli_epi16(y, 4);
        __m128i y_h = _mm_and_si128(y, m_mask_l_odds);
        __m128i y_wh = _mm_shuffle_epi8(m_corr_lut, y_h);

        // Add together and add horizontally
        __m128i x_w = _mm_sad_epu8(_mm_add_epi8(x_wl, x_wh), _mm_setzero_si128());
        x_w = _mm_add_epi32(x_w, _mm_unpackhi_epi64(x_w, _mm_setzero_si128()));
        __m128i y_w = _mm_sad_epu8(_mm_add_epi8(y_wl, y_wh), _mm_setzero_si128());
        y_w = _mm_add_epi32(y_w, _mm_unpackhi_epi64(y_w, _mm_setzero_si128()));

        // Accumulate and subtract bias
        ullWeight += _mm_cvtsi128_si32(x_w);
        ullWeightS += _mm_cvtsi128_si32(y_w);

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

    __m128i z = _mm_slli_epi64(y, 1);
    __m128i z1 = _mm_slli_si128(y, 8);
    z1 = _mm_srli_epi64(z1, 63);
    z = _mm_or_si128(z, z1);

    x = _mm_xor_si128(x, y);
    y = _mm_xor_si128(y, z);

    __m128i x_l = _mm_and_si128(x, m_mask_l_odds);
    __m128i y_l = _mm_and_si128(y, m_mask_l_odds);

    __m128i x_wl = _mm_shuffle_epi8(m_corr_lut, x_l);
    __m128i y_wl = _mm_shuffle_epi8(m_corr_lut, y_l);

    x = _mm_srli_epi16(x, 4);
    __m128i x_h = _mm_and_si128(x, m_mask_l_odds);
    __m128i x_wh = _mm_shuffle_epi8(m_corr_lut, x_h);
    y = _mm_srli_epi16(y, 4);
    __m128i y_h = _mm_and_si128(y, m_mask_l_odds);
    __m128i y_wh = _mm_shuffle_epi8(m_corr_lut, y_h);

    __m128i x_w = _mm_sad_epu8(_mm_add_epi8(x_wl, x_wh), _mm_setzero_si128());
    x_w = _mm_add_epi32(x_w, _mm_unpackhi_epi64(x_w, _mm_setzero_si128()));
    __m128i y_w = _mm_sad_epu8(_mm_add_epi8(y_wl, y_wh), _mm_setzero_si128());
    y_w = _mm_add_epi32(y_w, _mm_unpackhi_epi64(y_w, _mm_setzero_si128()));

    ullWeight += _mm_cvtsi128_si32(x_w);
    ullWeightS += _mm_cvtsi128_si32(y_w);

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

#define MIN(a,b) (((a)<(b))?(a):(b))

int main(int argc, char *argv[])
{
    // Manchester test
    static const size_t x_sz = 1024 * 1024 * 512;
    //uint8_t x[10] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    //uint8_t x[10] = {0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0x55, 0xBC};
    uint8_t *x = malloc(x_sz);
    uint8_t *x_man = malloc(2 * x_sz);
    uint8_t *x_man_sse = malloc(2 * x_sz);
    uint8_t *x_man_s = malloc(2 * x_sz);
    uint8_t *x_dec = malloc(x_sz);
    uint8_t *x_dec_sse = malloc(x_sz);
    uint8_t *x_dec_s = malloc(x_sz);
    uint8_t *x_dec_s_sse = malloc(x_sz);
    uint8_t x_aligned;
    uint8_t pass = 1;
    double elapsed_time;
    clock_t start_time;

    srand((unsigned)time(NULL));
    for(size_t i = 0; i < x_sz; i++)
        x[i] = rand() % UINT8_MAX;

    printf("x\t\t\t- [");
    for(size_t i = 0; i < MIN(x_sz, 10); i++)
        printf("%02X ", x[i]);
    printf("... ");
    for(size_t i = MIN(x_sz, 10); i > 0; i--)
        printf("%02X ", x[x_sz - i]);
    printf("]\r\n");

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

    memset(x_man_sse, 0, 2*x_sz);
    start_time = clock();
    manchester_encode_ssse3(x, x_sz, x_man_sse);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    printf("x_man_ssse3\t\t- [");
    for(size_t i = 0; i < MIN(2*x_sz, 10); i++)
        printf("%02X ", x_man_sse[i]);
    printf("... ");
    for(size_t i = MIN(2*x_sz, 10); i > 0; i--)
        printf("%02X ", x_man_sse[x_sz - i]);
    printf("]\r\n");

    printf("SSSE3 Manchester encode: %.3f ms (%.3f Mbps)\r\n", elapsed_time * 1000.0, x_sz * 8 / elapsed_time / 1e6);

    for(size_t i = 0; i < 2*x_sz; i++)
    {
        if(x_man_sse[i] != x_man[i])
        {
            pass = 0;

            break;
        }
    }

    printf("Manchester encode - SSSE3 vs Soft: %s\r\n", pass ? "PASSED" : "FAILED !!!!!!!!!!");

    shift_left(x_man, 2*x_sz, x_man_s);

    printf("x_man_s\t\t\t- [");
    for(size_t i = 0; i < MIN(2*x_sz, 10); i++)
        printf("%02X ", x_man_s[i]);
    printf("... ");
    for(size_t i = MIN(2*x_sz, 10); i > 0; i--)
        printf("%02X ", x_man_s[x_sz - i]);
    printf("]\r\n");

    x_aligned = 0;

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

    x_aligned = 0;

    memset(x_dec_sse, 0, x_sz);
    start_time = clock();
    manchester_decode_ssse3(x_man, 2*x_sz, x_dec_sse, &x_aligned);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    printf("x_dec_ssse3 (%hhu)\t\t- [", x_aligned);
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

    x_aligned = 0;

    memset(x_dec, 0, x_sz);
    start_time = clock();
    manchester_decode_ssse3(x_man_s, 2*x_sz, x_dec, &x_aligned);
    elapsed_time = (double)(clock() - start_time) / CLOCKS_PER_SEC;

    shift_right(x_dec, x_sz, x_dec_s_sse);

    printf("x_dec_s_ssse3 (%hhu)\t- [", x_aligned);
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

    return 0;
}
// SPDX-License-Identifier: Apache-2.0

// Add an amount to a given pixel
void brighten(float4 amount, inout float4 pixel)
{
    pixel += amount;
}

inline float linearToSRGB(float lin)
{
    if (lin <= 0.0031308f)
    {
        return lin * 12.92f;
    }
    else
    {
        return pow(lin, (1.0f / 2.4f)) * (1.055f) - 0.055f;
    }
}

/** Returns a sRGB version of an input linear RGB color in the ITU-R BT.709 color space
    \param LinearColor linear input color
*/
inline float3 linearToSRGB(float3 lin)
{
    return float3(
        linearToSRGB(lin.x),
        linearToSRGB(lin.y),
        linearToSRGB(lin.z));
}


void aces_filmic(in float4 input, out float4 output)
{
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;

    float3 color = input.xyz;
    color = color / (1.0 + color);
    output = float4(linearToSRGB(color), 1.0f);
}
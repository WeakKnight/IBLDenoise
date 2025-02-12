static const float MinimumAlpha = 0.08 * 0.08;
static const float M_PI = 3.14159265;
static const float M_1_PI = 0.318309886183790671538; // 1/pi

#define FP32_NaN (asfloat(0x7fc00000))

float linearToSRGB(float lin)
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
float3 linearToSRGB(float3 lin)
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

/*
https://graphics.pixar.com/library/OrthonormalB/paper.pdf
*/
void branchlessONB(float3 n, out float3 b1, out float3 b2)
{
    float sign = n.z >= 0.0f ? 1.0f : -1.0f;
    float a = -1.0f / (sign + n.z);
    float b = n.x * n.y * a;
    b1 = float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = float3(b, sign + n.y * n.y * a, -n.y);
}

// ---- Coordinate Frame
struct Frame
{
    float3 normal;
    float3 tangent;
    float3 bitangent;

    float3 toWorld(float3 v)
    {
        return v.x * tangent + v.y * bitangent + v.z * normal;
    }

    float3 toLocal(float3 v)
    {
        return float3(dot(v, tangent), dot(v, bitangent), dot(v, normal));
    }
};

Frame buildCoordinateFrame(float3 N)
{
    Frame f;
    f.normal = N;
    branchlessONB(N, f.tangent, f.bitangent);
    return f;
}

float Frame_cosTheta(float3 v)
{
    return v.z;
}

float Frame_absCosTheta(float3 v)
{
    return abs(v.z);
}

float3 utFaceForward(float3 v, float3 n)
{
    return (dot(v, n) < 0.f) ? -v : v;
}

float utHeaviside(float x)
{
    return step(1e-6, x);
}

float3 utReflect(float3 v, float3 n)
{
    return -v + 2 * dot(v, n) * n;
}

float NDF_GGX(float alpha, float NoH)
{
#if 0
    float a2 = alpha * alpha;
    float_t d = ((NoH * a2 - NoH) * NoH + 1);
    return min(a2 / (UNITY_PI * d * d), FP16_MAX);
#else
    NoH = min(NoH, float(0.99951172));
    float a2 = alpha * alpha;
    float d = ((NoH * a2 - NoH) * NoH + 1);
    float t = alpha / d;
    return t * t * M_1_PI;
#endif
}

float NDF_GGX_Anisotropic(float at, float ab, float NdotH, float TdotH, float BdotH)
{
    // D = 1 / (M_PI * at * ab * cosTheta^4 * (1 + tanTheta^2 * (cosPhi^2 / at^2 + sinPhi^2 / ab^2))^2)
    //   = 1 / (M_PI * at * ab * (cosTheta^2 + sinTheta^2 * (cosPhi^2 / at^2 + sinPhi^2 / ab^2))^2)
    //   = 1 / (M_PI * at * ab * (cosTheta^2 + (sinTheta * cosPhi / at)^2 + (sinTheta * sinPhi / ab)^2)^2)
    // plugin into following variable NdotH = cosTheta, TdotH = sinTheta * cosPhi, BdotH = sinTheta * sinPhi we got
    // D = 1 / (M_PI * at * ab * (NdotH^2 + (TdotH / at)^2 + (BdotH / ab)^2)^2)
    //   = (at * ab)^3 / (M_PI * (at * ab)^4 * (NdotH^2 + (TdotH / at)^2 + (BdotH / ab)^2)^2)
    //   = (at * ab)^3 / (M_PI * ((NdotH * at * ab)^2 + (TdotH * ab)^2 + (BdotH * at)^2)^2)
    // let a2 = at * ab
    // D = a2^3 / (M_PI * ((NdotH * a2)^2 + (TdotH * ab)^2 + (BdotH * at)^2)^2)
    float a2 = at * ab;
    float3 d = float3(a2 * NdotH, ab * TdotH, at * BdotH);
    float d2 = dot(d, d);
    float b2 = a2 / d2;
    return a2 * b2 * b2 * (1.0 / M_PI);
}

float smithMaskingFunction(float NdotV, float alpha)
{
    float denom = 1 + sqrt(1 + alpha * alpha * (1 - NdotV * NdotV) / (NdotV * NdotV));
    return 2.0 / denom;
}

float sqr(float x) { return x * x; }

float smithAnisotropicMaskingFunction(float NdotV, float TdotV, float BdotV, float at, float ab)
{
    return 2 * NdotV / (NdotV + sqrt(sqr(TdotV * at) + sqr(BdotV * ab) + sqr(NdotV)));
}

// TODO: @xiaoxinguo - condider to use height correlated masking and shadowing function???
// We made assumption about separable masking and shadowing function in ImportanceSampling.cginc regarding sample weight of visible NDF
// please correct it is we change to correlated masking shadowing function
float G_GGX(float NdotV, float NdotL, float alpha)
{
    return smithMaskingFunction(NdotV, alpha) * smithMaskingFunction(NdotL, alpha);
}

float G_GGX_Anisotropic(float NdotV, float TdotV, float BdotV,
                       float NdotL, float TdotL, float BdotL,
                       float at, float ab)
{
    return smithAnisotropicMaskingFunction(NdotV, TdotV, BdotV, at, ab) * smithAnisotropicMaskingFunction(NdotL, TdotL, BdotL, at, ab);
}

// Refer to http://jcgt.org/published/0007/04/01/
// Input Ve: view direction
// Input alpha_x, alpha_y: roughness parameters
// Input U1, U2: uniform random numbers
// Output Ne: normal sampled with PDF D_Ve(Ne) = G1(Ve) * max(0, dot(Ve, Ne)) * D(Ne) / Ve.z
float3 _sampleGGXVNDF(float3 Ve, float alpha_x, float alpha_y, float U1, float U2)
{
    // Section 3.2: transforming the view direction to the hemisphere configuration
    float3 Vh = normalize(float3(alpha_x * Ve.x, alpha_y * Ve.y, Ve.z));
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    float3 T1 = lensq > 0 ? float3(-Vh.y, Vh.x, 0) * rsqrt(lensq) : float3(1, 0, 0);
    float3 T2 = cross(Vh, T1);
    // Section 4.2: parameterization of the projected area
    float r = sqrt(U1);
    float phi = 2.0 * M_PI * U2;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;
    // Section 4.3: reprojection onto hemisphere
    float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    float3 Ne = normalize(float3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0, Nh.z)));
    return Ne;
}

/*
 * GGX distribution
 */
float3 squareToGgxDistribution(float2 u, float alpha)
{
    float a2 = alpha * alpha;

    float phi = M_PI * 2 * u.x;
    float cosTheta = sqrt(max(0, (1 - u.y)) / (1 + (a2 - 1) * u.y));
    float sinTheta = sqrt(max(0, 1 - cosTheta * cosTheta));
    return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// GGX microfacet distribution
struct MicrofacetDistribution
{
    bool sampleVisible;
    bool isotropic;
    float alpha_x;
    float alpha_y;

    bool effectivelySmooth()
    {
        return max(alpha_x, alpha_y) < 1e-3f;
    }

    float D(float3 wh)
    {
        if (isotropic)
            return NDF_GGX(alpha_x, Frame_cosTheta(wh));
        else
            return NDF_GGX_Anisotropic(alpha_x, alpha_y, wh.z, wh.x, wh.y);
    }

    float G1(float3 w)
    {
        if (isotropic)
            return smithMaskingFunction(Frame_cosTheta(w), alpha_x);
        else
            return smithAnisotropicMaskingFunction(w.z, w.x, w.y, alpha_x, alpha_y);
    }

    float G(float3 wo, float3 wi)
    {
        if (isotropic)
            return G_GGX(Frame_cosTheta(wo), Frame_cosTheta(wi), alpha_x);
        else
            return G_GGX_Anisotropic(wo.z, wo.x, wo.y,
                                     wi.z, wi.x, wi.y,
                                     alpha_x, alpha_y);
    }

    float3 sample_wh(float3 wo, float2 u)
    {
        if (sampleVisible)
        {
            return _sampleGGXVNDF(utFaceForward(wo, float3(0, 0, 1)), alpha_x, alpha_y, u.x, u.y);
        }
        else
        {
            // TODO: support anisotropic GGX sampling
            if (isotropic)
                return squareToGgxDistribution(u, alpha_x);
            else
                return FP32_NaN;
        }
    }

    float pdf(float3 wo, float3 wh)
    {
        if (sampleVisible)
            return D(wh) * G1(wo) * abs(dot(wo, wh)) / Frame_absCosTheta(wo);
        else
            return D(wh) * Frame_absCosTheta(wh);
    }
};

struct BSDFSamplingRecord {
    /// Outgoing direction (in the local frame)
    float3 wo;

    /// Incident direction (in the local frame)
    float3 wi;
};

/// Create a new record for sampling the BSDF
BSDFSamplingRecord BSDFSamplingRecord_(in float3 wo_)
{
    BSDFSamplingRecord rec;
    rec.wo = wo_;
    return rec;
}

struct MicrofacetReflection
{
    MicrofacetDistribution distr;

    float sample(inout BSDFSamplingRecord bRec, in float2 sample_)
    {
        float3 wh = distr.sample_wh(bRec.wo, sample_);
        wh *= sign(bRec.wo.z);

        float3 wi = utReflect(bRec.wo, wh);
        if (Frame_cosTheta(bRec.wo) * Frame_cosTheta(wi) <= 0)
        {
            bRec.wi = 0;
            return 0;
        }

        bRec.wi = wi;

        if (distr.sampleVisible)
        {
            float cosThetaI = Frame_cosTheta(bRec.wi);
            return distr.G(bRec.wo, bRec.wi) / abs(distr.G1(bRec.wo) * cosThetaI);
        }
        else
        {
            float cosThetaO = Frame_cosTheta(bRec.wo);
            float cosThetaI = Frame_cosTheta(bRec.wi);
            float cosThetaH = Frame_cosTheta(wh);
            return distr.G(bRec.wo, bRec.wi) * dot(bRec.wi, wh) / abs(cosThetaO * cosThetaI * cosThetaH);
        }
    }
};

MicrofacetDistribution MicrofacetDistribution_(bool sampleVisible_, float alpha_)
{
    MicrofacetDistribution distr;
    distr.sampleVisible = sampleVisible_;
    distr.isotropic = true;
    distr.alpha_x = alpha_;
    distr.alpha_y = alpha_;
    return distr;
}

MicrofacetReflection MicrofacetReflection_(bool sampleVisible_, float alpha_)
{
    MicrofacetReflection bsdf;
    bsdf.distr = MicrofacetDistribution_(sampleVisible_, alpha_);
    return bsdf;
}

// QMC sequence from http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
// Working well for low dimension with high sample count (tested 4 dimensions) https://github.com/anderslanglands/notebooks/blob/master/r2_sequence_01.ipynb
static float2 s_R2_alpha = float2(0.7548776662466927, 0.5698402909980532);
static float4 s_R4_alpha = float4(0.8566748838545029, 0.733891856627126, 0.6287067210378086, 0.53859725722361);

// Be careful with numerical precision, as n increased it's recommend to use nextSampleR2Sequence instead
float2 nthSampleR2Sequence(int n)
{
#if 0
    return frac(0.5 + s_R2_alpha * (n + 1));
#else
    // MAD form
    return frac((0.5 + s_R2_alpha) + s_R2_alpha * n);
#endif
}

float2 nextSampleR2Sequence(float2 s)
{
    return frac(s_R2_alpha + s);
}

// Based on: [Clarberg 2008, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD"]
// Removed branch before division by using a small epsilon
// https://fileadmin.cs.lth.se/graphics/research/papers/2008/simdmapping/clarberg_simdmapping08_preprint.pdf
float2 inverseEqualAreaSphericalMapping(float3 Direction)
{
    // Most use cases of this func generate Direction by diffing two positions and thus unnormalized
    Direction = normalize(Direction);

    float3 AbsDir = abs(Direction);
    float R = sqrt(1 - AbsDir.z);
    float Epsilon = 5.42101086243e-20; // 2^-64 (this avoids 0/0 without changing the rest of the mapping)
    float x = min(AbsDir.x, AbsDir.y) / (max(AbsDir.x, AbsDir.y) + Epsilon);

    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi, x=[0,1].
    const float t1 = 0.406758566246788489601959989e-5f;
    const float t2 = 0.636226545274016134946890922156f;
    const float t3 = 0.61572017898280213493197203466e-2f;
    const float t4 = -0.247333733281268944196501420480f;
    const float t5 = 0.881770664775316294736387951347e-1f;
    const float t6 = 0.419038818029165735901852432784e-1f;
    const float t7 = -0.251390972343483509333252996350e-1f;

    // Polynomial approximation of atan(x)*2/pi
    float Phi = t6 + t7 * x;
    Phi = t5 + Phi * x;
    Phi = t4 + Phi * x;
    Phi = t3 + Phi * x;
    Phi = t2 + Phi * x;
    Phi = t1 + Phi * x;

    Phi = (AbsDir.x < AbsDir.y) ? 1 - Phi : Phi;
    float2 UV = float2(R - Phi * R, Phi * R);
    UV = (Direction.z < 0) ? 1 - UV.yx : UV;
    UV = asfloat(asuint(UV) ^ (asuint(Direction.xy) & 0x80000000u));
    return UV * 0.5 + 0.5;
}

// Wrap around octahedral map for correct hardware bilinear filtering
uint2 octahedralMapWrapBorder(uint2 TexelCoord, uint Resolution, uint BorderSize)
{
    if (TexelCoord.x < BorderSize)
    {
        TexelCoord.x = BorderSize - 1 + BorderSize - TexelCoord.x;
        TexelCoord.y = Resolution - 1 - TexelCoord.y;
    }
    if (TexelCoord.x >= Resolution - BorderSize)
    {
        TexelCoord.x = (Resolution - BorderSize) - (TexelCoord.x - (Resolution - BorderSize - 1));
        TexelCoord.y = Resolution - 1 - TexelCoord.y;
    }
    if (TexelCoord.y < BorderSize)
    {
        TexelCoord.y = BorderSize - 1 + BorderSize - TexelCoord.y;
        TexelCoord.x = Resolution - 1 - TexelCoord.x;
    }
    if (TexelCoord.y >= Resolution - BorderSize)
    {
        TexelCoord.y = (Resolution - BorderSize) - (TexelCoord.y - (Resolution - BorderSize - 1));
        TexelCoord.x = Resolution - 1 - TexelCoord.x;
    }

    return TexelCoord - BorderSize;
}

// Based on: [Clarberg 2008, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD"]
// Fixed sign bit for UV.y == 0 and removed branch before division by using a small epsilon
// https://fileadmin.cs.lth.se/graphics/research/papers/2008/simdmapping/clarberg_simdmapping08_preprint.pdf
float3 equalAreaSphericalMapping(float2 UV)
{
    UV = 2 * UV - 1;
    float D = 1 - (abs(UV.x) + abs(UV.y));
    float R = 1 - abs(D);
    // Branch to avoid dividing by 0.
    // Only happens with (0.5, 0.5), usually occurs in odd number resolutions which use the very central texel
    float Phi = R == 0 ? 0 : (M_PI / 4) * ((abs(UV.y) - abs(UV.x)) / R + 1);
    float F = R * sqrt(2 - R * R);
    return float3(
        F * sign(UV.x) * abs(cos(Phi)),
        F * sign(UV.y) * abs(sin(Phi)),
        sign(D) * (1 - R * R)
    );
}
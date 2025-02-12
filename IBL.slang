import Common;

static const uint _BorderSize = 1;
static const uint _ProbeSize = 256;

int GetProbeSize(int mipLevel)
{
    return _ProbeSize >> mipLevel;
}

float4 _PrefilterEnvmap(float3 R, float alpha, 
    Texture2D<float4> _RadianceMap, int SampleCount)
{
    alpha = max(MinimumAlpha, alpha);
    
    float3 V = R;
    float3 N = R;

    Frame frame = buildCoordinateFrame(N);
    MicrofacetReflection bsdf = MicrofacetReflection_(true, alpha);
    
    float3 accumulateValue = 0;
    float accumulateWeight = 0;
    float2 u = nthSampleR2Sequence(0);
    for (int i = 0; i < SampleCount; i++)
    {
        BSDFSamplingRecord bRec = BSDFSamplingRecord_(frame.toLocal(V));
        float bsdfWeight = bsdf.sample(bRec, u);

        if (Frame_cosTheta(bRec.wi) > 1e-5f && bsdfWeight > 1e-5f)
        {
            float3 lightDirection = frame.toWorld(bRec.wi);
            float2 uv = inverseEqualAreaSphericalMapping(lightDirection);

            uint2 texel = uint2(uv * (_ProbeSize - 2 * _BorderSize)) + _BorderSize;
            float3 Li = _RadianceMap[texel].xyz;

            accumulateValue += Li * Frame_cosTheta(bRec.wi) * bsdfWeight;
            accumulateWeight += Frame_cosTheta(bRec.wi) * bsdfWeight;
        }

        u = nextSampleR2Sequence(u);
    }
    
    return float4(accumulateValue, accumulateWeight);    
}

float4 PrefilterEnvmap(uint2 threadIndex,
          float _Alpha,
          int _MipLevel,
          int _SampleCount,
          Texture2D<float4> radianceMap)
{
    uint borderSize = _BorderSize;
    uint2 octahedralBorderlessCoord = octahedralMapWrapBorder(threadIndex, GetProbeSize(_MipLevel), borderSize);
    float2 uv = (octahedralBorderlessCoord + 0.5f) / (GetProbeSize(_MipLevel) - 2 * borderSize);
    float3 R = equalAreaSphericalMapping(uv);
    float4 x = _PrefilterEnvmap(R, _Alpha, radianceMap, _SampleCount);
    return float4(x.xyz / max(x.w, 1e-5f), 1.0f);
}

/*
    0.05 0.15 0.05
    0.15 0.2  0.15
    0.05 0.15 0.05
*/
float4 BoxBlur(uint2 threadIndex, Texture2D<float4> prefilteredRadianceMap, uint w, uint h)
{
    float3 kernelWeights[9] =
    {
        float3(-1.0f, -1.0f, 0.05f),
        float3(-1.0f, 1.0f, 0.05f),
        float3(1.0f, -1.0f, 0.05f),
        float3(1.0f, 1.0f, 0.05f),
        float3(0.0f, 1.0f, 0.15f),
        float3(0.0f, -1.0f, 0.15f),
        float3(1.0f, 0.0f, 0.15f),
        float3(-1.0f, 0.0f, 0.15f),
        float3(0.0f, 0.0f, 0.2f),
    };

    float3 res = 0;
    float totalWeight = 0;
    for (int i = 0; i < 9; i++)
    {
        float3 k = kernelWeights[i];
        int2 offset = int2(k.xy);
        float weight = k.z;
        
        int2 targetIndex = (int2)(threadIndex) + offset;
        targetIndex.x = clamp(targetIndex.x, 0, int(w) - 1);
        targetIndex.y = clamp(targetIndex.y, 0, int(h) - 1);

        float3 col = prefilteredRadianceMap.Load(int3(targetIndex, 0)).xyz;
        res += col * weight;
        totalWeight += weight;
    }

    return float4(res / totalWeight, 1.0);
}
import Common;

static const uint _BorderSize = 1;
static const uint _ProbeSize = 256;
static const uint sampleCount = 1024;

int GetProbeSize(int mipLevel)
{
    return _ProbeSize >> mipLevel;
}

float4 _PrefilterEnvmap(float3 R, float alpha, 
    Texture2D<float4> _RadianceMap)
{
    alpha = max(MinimumAlpha, alpha);
    
    float3 V = R;
    float3 N = R;

    Frame frame = buildCoordinateFrame(N);
    MicrofacetReflection bsdf = MicrofacetReflection_(true, alpha);
    
    float3 accumulateValue = 0;
    float accumulateWeight = 0;
    float2 u = nthSampleR2Sequence(0);
    for (uint i = 0; i < sampleCount; i++)
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
          Texture2D<float4> radianceMap)
{
    uint borderSize = _BorderSize;
    uint2 octahedralBorderlessCoord = octahedralMapWrapBorder(threadIndex, GetProbeSize(_MipLevel), borderSize);
    float2 uv = (octahedralBorderlessCoord + 0.5f) / (GetProbeSize(_MipLevel) - 2 * borderSize);
    float3 R = equalAreaSphericalMapping(uv);
    float4 x = _PrefilterEnvmap(R, _Alpha, radianceMap);
    return float4(x.xyz / max(x.w, 1e-5f), 1.0f);
}
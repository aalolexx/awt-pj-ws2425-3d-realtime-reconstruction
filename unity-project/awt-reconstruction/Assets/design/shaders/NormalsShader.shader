Shader "Custom/NormalsShader"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _CullMode ("Cull Mode", Float) = 0 // 0: Off, 1: Front, 2: Back
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            Cull [_CullMode]

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata_t
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };

            struct v2f
            {
                float3 normal : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata_t v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.normal = normalize(mul((float3x3)unity_WorldToObject, v.normal));
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                return fixed4(i.normal * 0.5 + 0.5, 1.0); // Map [-1,1] range to [0,1] range
            }
            ENDCG
        }
    }
    FallBack "Diffuse"
}
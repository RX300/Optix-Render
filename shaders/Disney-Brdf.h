#pragma once
#include"intersection.h"
#include"rand_gen.h"
#include"shader_common.h"
namespace Disney_Brdf {
	__forceinline__ __device__ float lerp(const float a, const float b, const float t)
	{
		return a + t * (b - a);
	}
	__forceinline__ __device__  float test(Material::disney_material* mat) { float gloss = lerp(0.1f, 0.001f, mat->clearcoatGloss); }
	__forceinline__ __device__  float sqr(const float& x) { return x * x; }
	__forceinline__ __device__ float3 sphericalDirection(float sinTheta, float cosTheta, float sinPhi, float cosPhi) 
	{
		return make_float3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
	}
	__forceinline__ __device__ float3 mon2lin(float3 x)
	{
		return make_float3(pow(x.x, 2.2f), pow(x.y, 2.2f), pow(x.z, 2.2f));
	}
	__forceinline__ __device__ float SchlickFresnel(const float& u)
	{
		float m = clamp(1 - u, 0.0f, 1.0f);
		float m2 = m * m;
		return m2 * m2 * m; // pow(m,5)
	}
	__forceinline__ __device__ float GTR1(float NdotH, float a)
	{
		if (a >= 1) return 1.0f / M_PIf;
		float a2 = a * a;
		float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
		return (a2 - 1.0f) / (M_PIf * log(a2) * t);
	}
	__forceinline__ __device__ float GTR2(float NdotH, float a)
	{
		float a2 = a * a;
		float t = 1.0f + (a2 - 1.0f) * NdotH * NdotH;
		return a2 / (M_PIf * t * t);
	}

	__forceinline__ __device__ float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
	{
		return 1.0f / (M_PIf * ax * ay * sqr(sqr(HdotX / ax) + sqr(HdotY / ay) + NdotH * NdotH));
	}
	__forceinline__ __device__ float smithG_GGX(float NdotV, float alphaG)
	{
		float a = alphaG * alphaG;
		float b = NdotV * NdotV;
		return 1.0f / (NdotV + sqrt(a + b - a * b));
	}
	__forceinline__ __device__ float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
	{
		return 1.0f / (NdotV + sqrt(sqr(VdotX * ax) + sqr(VdotY * ay) + sqr(NdotV)));
	}
	__forceinline__ __device__ float3 sampleDisneyDiffuse(const Material::intersection& its, const float2& rand2,const float3& wo, const float3& wi, Material::disney_material* mat)
	{
		return its.tbn.transformToWorld(Shader_COM::sampleCosineWeightedHemisphere(rand2));
	}
	__forceinline__ __device__ float3 sampleDisneySpecular(const Material::intersection& its,const float2& rand2,const float3& wo, const float3& wi, Material::disney_material* mat)
	{
		float cosTheta = 0.0f, phi = 0.0f;

		float aspect = sqrt(1.0f - mat->anisotropic * 0.9f);
		float alphax = fmaxf(0.001f, sqr(mat->roughness) / aspect);
		float alphay = fmaxf(0.001f, sqr(mat->roughness) * aspect);
		phi = atan(alphay / alphax * tan(2.0f * M_PIf * rand2.y + 0.5f * M_PIf));

		if (rand2.y > 0.5f) phi += M_PIf;
		float sinPhi = sin(phi), cosPhi = cos(phi);
		float alphax2 = alphax * alphax, alphay2 = alphay * alphay;
		float alpha2 = 1.0f / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
		float tanTheta2 = alpha2 *rand2.x / (1.0f - rand2.x);
		cosTheta = 1.0f / sqrt(1.0f + tanTheta2);

		float sinTheta = sqrt(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
		float3 whLocal = sphericalDirection(sinTheta, cosTheta, sin(phi), cos(phi));

		float3 wh = its.tbn.transformToWorld(whLocal);

		if (!Shader_COM::sameHemiSphere(wo, wh, its.normal)) {
			wh *= -1.0f;
		}

		return  reflect(-wo, wh);
	}
	__forceinline__ __device__ float3 sampleDisneyClearcoat(const Material::intersection& its, const float2& rand2, const float3& wo, const float3& wi, Material::disney_material* mat)
	{
		float gloss = lerp(0.1f, 0.001f, mat->clearcoatGloss);
		float alpha2 = gloss * gloss;
		float cosTheta = sqrt(fmaxf(EPSILON, float((1. - pow(alpha2, 1. - rand2.x)) / (1. - alpha2))));
		float sinTheta = sqrt(fmaxf(EPSILON, float(1. - cosTheta * cosTheta)));
		float phi = M_2PIf * rand2.y;

		float3 whLocal = sphericalDirection(sinTheta, cosTheta, sin(phi), cos(phi));
		float3 wh = its.tbn.transformToWorld(whLocal);

		if (!Shader_COM::sameHemiSphere(wo, wh, its.normal)) {
			wh *= -1.;
		}

		return  reflect(-wo, wh);
	}
	__forceinline__ __device__ float DisneyDiffusePdf(const Material::intersection& its, const float3& wo, const float3& wi, Material::disney_material* mat)
	{
		return dot(wi, its.normal) / M_PIf;
	}
	__forceinline__ __device__ float  DisneySpecularPdf(const Material::intersection& its, const float3& wo, const float3& wi, Material::disney_material* mat)
	{
		if (!Shader_COM::sameHemiSphere(wo, wi, its.normal)) return 0.0f;
		float3 wh = normalize(wo + wi);
		float3 X = its.tangent;
		float3 Y = its.bitangent;

		float aspect = sqrt(1.0f - mat->anisotropic * 0.9f);
		float alphax = fmaxf(0.001f, sqr(mat->roughness) / aspect);
		float alphay = fmaxf(0.001f, sqr(mat->roughness) * aspect);

		float alphax2 = alphax * alphax;
		float alphay2 = alphax * alphay;

		float hDotX = dot(wh, X);
		float hDotY = dot(wh, Y);
		float NdotH = dot(its.normal, wh);

		float denom = hDotX * hDotX / alphax2 + hDotY * hDotY / alphay2 + NdotH * NdotH;
		if (denom == 0.) return 0.;
		float pdfDistribution = NdotH / (M_PIf * alphax * alphay * denom * denom);
		return pdfDistribution / (4. * dot(wo, wh));
	}
	__forceinline__ __device__ float  DisneyClearcoatPdf(const Material::intersection& its, const float3& wo, const float3& wi, Material::disney_material* mat)
	{
		if (!Shader_COM::sameHemiSphere(wo, wi, its.normal)) return 0.0f;

		float3 wh = normalize(wo + wi);

		float NdotH = abs(dot(wh, its.normal));
		float Dr = GTR1(NdotH, lerp(.1f, .001f, mat->clearcoatGloss));
		return Dr * NdotH / (4. * dot(wo, wh));
	}
	__forceinline__ __device__ float3 evalBxdf(const Material::intersection& its, const float3& wo, const float3& wi, Material::disney_material* mat)
	{
		//׼������
		float3 L = normalize(wi); float3 V = normalize(wo); float3 N = normalize(its.normal); float3 X = normalize(its.tangent); float3 Y = normalize(its.bitangent);
		float NdotL = dot(N, wi);
		float NdotV = dot(N, wo);
		if (NdotL < 0.0f || NdotV < 0.0f) return make_float3(0.0f); //����ˮƽ�����µĹ��߻�����
		float3 H = normalize(L + V);                    //�������
		float NdotH = dot(N, H);
		float LdotH = dot(L, H);

		float3 basecolor_temp;
		if (mat->basecolor_tex != NULL)
		{
			float4 temp = tex2D<float4>(mat->basecolor_tex, its.uv_coord.x, its.uv_coord.y);
			basecolor_temp = make_float3(temp.x, temp.y, temp.z);
		}
		else
			basecolor_temp = mat->baseColor;


		float3 Cdlin = mon2lin(basecolor_temp); //��gamma�ռ����ɫת�������Կռ䣬Ŀǰ�洢�Ļ���rgb
		float Cdlum = dot(Cdlin, make_float3(0.3f, 0.6f, 0.1f)); //luminance approx.���ƵĽ�rgbת���ɹ����� luminance
		//��baseColor�����ȹ�һ�����Ӷ�������ɫ���ͱ��Ͷȣ�������ΪCtint���������޹صĹ���ɫ��
		float3 Ctint = Cdlum > 0.0f ? (Cdlin / Cdlum) : make_float3(1.0f);
		//�����õ��߹��ɫ������2��mix����
		//��һ�δӰ�ɫ�����û������specularTint��ֵ��Ctint������specularTintΪ0���򷵻ش���ɫ
		//�ڶ��δӵ�һ�λ����ķ���ֵ��ʼ�����ս�����metallic����ֵ���������ȵ����Կռ�baseColor��
		//���������Ϊ1���򷵻ر�ɫbaseColor�����������Ϊ0���ȵ���ʣ���ô���ص�һ�β�ֵ�Ľ����һ��������0.8 * specular����
		float3 Cspec0 = lerp(mat->specular * 0.08f * lerp(make_float3(1.0f), Ctint, mat->specularTint), Cdlin, mat->metallic);
		//���ǹ�����ɫ������֪����������ڲ������ϵȲ�����FresnelPeak���Ķ�����ܣ�������ɫ��Ӱ�ɫ��ʼ�������û������sheenTintֵ����ֵ��CtintΪֹ��
		float3 Csheen = lerp(make_float3(1.0f), Ctint, mat->sheenTint);

		//���´���θ������Diffuse����
		// Diffuse fresnel - go from 1 at normal incidence to 0.5 at grazing
		// and glm::mix in diffuse retro-reflection based on roughness
		float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV); //SchlickFresnel���ص���(1 - cos��) ^ 5�ļ�����
		float Fd90 = 0.5f + 2.0f * LdotH * LdotH * mat->roughness; //ʹ�ôֲڶȼ���������ķ�����
		float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);  //�˲��軹û�г���baseColor / pi�����ڵ�ǰ�����֮����ɡ�

		//���´��븺�����SS(�α���ɢ��)����
		// Based on Hanrahan - Krueger brdf approximation of isotropic bssrdf�����ڸ���ͬ��bssrdf��Hanrahan - Krueger brdf�ƽ���
		//1.25 scale is used to(roughly) preserve albedo��1.25�����������ڣ����£����������ʣ�
		// Fss90 used to "flatten" retroreflection based on roughness ��Fss90���ڡ�ƽ�������ڴֲڶȵ��淴�䣩
		float Fss90 = LdotH * LdotH * mat->roughness; //��ֱ�ڴα���ķ�����ϵ��
		float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
		float ss = 1.25f * (Fss * (1.0f / (NdotL + NdotV) - .5f) + .5f); //�˲���ͬ����û�г���baseColor / pi�����ڵ�ǰ�����֮����ɡ�


		// Specular
		/*	// ���淴�� -- ����ͬ��
		float alpha = std::max(0.001f, sqr(roughness));
		float Ds = GTR2(NdotH, alpha);
		float FH = SchlickFresnel(LdotH);
		glm::vec3 Fs = mix(Cspec0, glm::vec3(1), FH);
		float Gs = smithG_GGX(NdotL, roughness);
		Gs *= smithG_GGX(NdotV, roughness);*/

		float aspect = sqrt(1.0f - mat->anisotropic * 0.9f); //aspect �����������е�anisotropic������ӳ�䵽[0.1, 1]�ռ䣬ȷ��aspect��Ϊ0,
		float ax = fmaxf(0.001f, sqr(mat->roughness) / aspect);                    //ax���Ų���anisotropic�����Ӷ�����
		float ay = fmaxf(0.001f, sqr(mat->roughness) * aspect);                    //ay���Ų���anisotropic�����Ӷ����٣�ax��ay��anisotropicֵΪ0ʱ���
		float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);  //��������GTR2������ӦH�ķ���ǿ��
		float FH = SchlickFresnel(LdotH);  //���ط�������ģ���pow(1 - cos��d, 5)
		float3 Fs = lerp(Cspec0, make_float3(1.0f), FH); //������ʹ����Cspec0��ΪF0���ȸ߹��ɫ��ģ������ķ�����Ⱦɫ
		float Gs;   //�����һ����l��v��n��أ���������ʱ����Ҫ���Ƿ���ռ��е�����t�͸�����b
		Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);  //�ڱι����ļ�����G1
		Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay); //��Ӱ�����ļ�����G1�����ϲ��������Gs

		// sheen �����������Ϊ��Ե��������Ĳ���
		float3 Fsheen = FH * mat->sheen * Csheen; //��ʿ����Ϊsheenֵ�����ڷ�������FH��ͬʱǿ�ȱ����Ʊ���sheen����ɫ���Ʊ���CsheenӰ��

		// clearcoat(ior = 1.5->F0 = 0.04)
		//�����û�������䣬ֻ�о��淴�䣬ʹ�ö�����D, F��G��
		//������ʹ��GTR1��berry���ֲ�������ȡ����ǿ�ȣ��ڶ���������a���ֲڶȣ�
		//��ʿ��ʹ���û����Ʊ���clearcoatGloss����0.1��0.001���Բ�ֵ��ȡa
		float Dr = GTR1(NdotH, lerp(0.1f, 0.001f, mat->clearcoatGloss));
		float Fr = lerp(0.04f, 1.0f, FH); //���������ϵ����ֵ��0.04
		float Gr = smithG_GGX(NdotL, 0.25f) * smithG_GGX(NdotV, .25f);   //������ʹ�ø���ͬ�Ե�smithG_GGX���㣬a�̶���0.25

		//�������� + ���� * �ǽ����� + ���淴�� + ����߹�
		// ע�����������ʱʹ����subsurface���Ʊ����Ի��ڷ������������� �� �α���ɢ����в�ֵ���ɣ����⻹������֮ǰ�ᵽ��baseColor / pi
		// ʹ�÷ǽ����ȣ��ȣ�1 - �����ȣ������������Խ����������� < -�����������ܲ�ֵ��ˬ��
		return ((1.0f / M_PIf) * lerp(Fd, ss, mat->subsurface) * Cdlin + Fsheen)
			* (1.0f - mat->metallic)
			+ Gs * Fs * Ds + 0.25f * mat->clearcoat * Gr * Fr * Dr;
	}

	__forceinline__ __device__ float bxdfPdf(const Material::intersection& its, const float3& wo, const float3& wi, Material::disney_material* mat)
	{
			// �ֱ�������� BRDF �ĸ����ܶ�
		float pdf_diffuse = DisneyDiffusePdf(its, wo,wi,mat );
		float pdf_specular = DisneySpecularPdf(its, wo, wi, mat);
		float pdf_clearcoat = DisneyClearcoatPdf(its, wo, wi, mat);
		// �����ͳ��
		float r_diffuse = (1.0 - mat->metallic);
		float r_specular = 1.0;
		float r_clearcoat = 0.25 * mat->clearcoat;
		float r_sum = r_diffuse + r_specular + r_clearcoat;

		// ���ݷ���ȼ���ѡ��ĳ�ֲ�����ʽ�ĸ���
		float p_diffuse = r_diffuse / r_sum;
		float p_specular = r_specular / r_sum;
		float p_clearcoat = r_clearcoat / r_sum;

		// ���ݸ��ʻ�� pdf
		float pdf = p_diffuse * pdf_diffuse
			+ p_specular * pdf_specular
			+ p_clearcoat * pdf_clearcoat;

		pdf = fmaxf(1e-10f, pdf);
		return pdf;

	}

	__forceinline__ __device__ float3 bxdfSample(const Material::intersection& its, const float3& wo, float3& wi, float& pdf, Material::disney_material* mat)
	{
		float2 rand2 = make_float2(curand_uniform(its.state), curand_uniform(its.state2));
		// �����ͳ��
		float r_diffuse = (1.0 - mat->metallic);
		float r_specular = 1.0;
		float r_clearcoat = 0.25 * mat->clearcoat;
		float r_sum = r_diffuse + r_specular + r_clearcoat;

		// ���ݷ���ȼ������
		float p_diffuse = r_diffuse / r_sum;
		float p_specular = r_specular / r_sum;
		float p_clearcoat = r_clearcoat / r_sum;

		float rd = curand_uniform(its.state);

		// ������
		if (rd <= p_diffuse) {
			wi = sampleDisneyDiffuse(its,rand2,wo,wi,mat);
		}
		// ���淴��
		else if (rd <= (p_diffuse + p_specular)) {
			wi = sampleDisneySpecular(its, rand2, wo, wi, mat);
		}
		// ����
		else {
			wi = sampleDisneyClearcoat(its, rand2, wo, wi, mat);
		}
		pdf = bxdfPdf(its,wo,wi,mat);
		return evalBxdf(its, wo, wi, mat);

	}
}
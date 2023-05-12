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
		//准备参数
		float3 L = normalize(wi); float3 V = normalize(wo); float3 N = normalize(its.normal); float3 X = normalize(its.tangent); float3 Y = normalize(its.bitangent);
		float NdotL = dot(N, wi);
		float NdotV = dot(N, wo);
		if (NdotL < 0.0f || NdotV < 0.0f) return make_float3(0.0f); //无视水平面以下的光线或视线
		float3 H = normalize(L + V);                    //半角向量
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


		float3 Cdlin = mon2lin(basecolor_temp); //将gamma空间的颜色转换到线性空间，目前存储的还是rgb
		float Cdlum = dot(Cdlin, make_float3(0.3f, 0.6f, 0.1f)); //luminance approx.近似的将rgb转换成光亮度 luminance
		//对baseColor按亮度归一化，从而独立出色调和饱和度，可以认为Ctint是与亮度无关的固有色调
		float3 Ctint = Cdlum > 0.0f ? (Cdlin / Cdlum) : make_float3(1.0f);
		//混淆得到高光底色，包含2次mix操作
		//第一次从白色按照用户输入的specularTint插值到Ctint。列如specularTint为0，则返回纯白色
		//第二次从第一次混淆的返回值开始，按照金属度metallic，插值到带有亮度的线性空间baseColor。
		//列如金属度为1，则返回本色baseColor。如果金属度为0，既电介质，那么返回第一次插值的结果得一定比例（0.8 * specular倍）
		float3 Cspec0 = lerp(mat->specular * 0.08f * lerp(make_float3(1.0f), Ctint, mat->specularTint), Cdlin, mat->metallic);
		//这是光泽颜色，我们知道光泽度用于补偿布料等材质在FresnelPeak处的额外光能，光泽颜色则从白色开始，按照用户输入的sheenTint值，插值到Ctint为止。
		float3 Csheen = lerp(make_float3(1.0f), Ctint, mat->sheenTint);

		//以下代码段负责计算Diffuse分量
		// Diffuse fresnel - go from 1 at normal incidence to 0.5 at grazing
		// and glm::mix in diffuse retro-reflection based on roughness
		float FL = SchlickFresnel(NdotL), FV = SchlickFresnel(NdotV); //SchlickFresnel返回的是(1 - cosθ) ^ 5的计算结果
		float Fd90 = 0.5f + 2.0f * LdotH * LdotH * mat->roughness; //使用粗糙度计算漫反射的反射率
		float Fd = lerp(1.0f, Fd90, FL) * lerp(1.0f, Fd90, FV);  //此步骤还没有乘以baseColor / pi，会在当前代码段之外完成。

		//以下代码负责计算SS(次表面散射)分量
		// Based on Hanrahan - Krueger brdf approximation of isotropic bssrdf（基于各向同性bssrdf的Hanrahan - Krueger brdf逼近）
		//1.25 scale is used to(roughly) preserve albedo（1.25的缩放是用于（大致）保留反照率）
		// Fss90 used to "flatten" retroreflection based on roughness （Fss90用于“平整”基于粗糙度的逆反射）
		float Fss90 = LdotH * LdotH * mat->roughness; //垂直于次表面的菲涅尔系数
		float Fss = lerp(1.0f, Fss90, FL) * lerp(1.0f, Fss90, FV);
		float ss = 1.25f * (Fss * (1.0f / (NdotL + NdotV) - .5f) + .5f); //此步骤同样还没有乘以baseColor / pi，会在当前代码段之外完成。


		// Specular
		/*	// 镜面反射 -- 各向同性
		float alpha = std::max(0.001f, sqr(roughness));
		float Ds = GTR2(NdotH, alpha);
		float FH = SchlickFresnel(LdotH);
		glm::vec3 Fs = mix(Cspec0, glm::vec3(1), FH);
		float Gs = smithG_GGX(NdotL, roughness);
		Gs *= smithG_GGX(NdotV, roughness);*/

		float aspect = sqrt(1.0f - mat->anisotropic * 0.9f); //aspect 将艺术家手中的anisotropic参数重映射到[0.1, 1]空间，确保aspect不为0,
		float ax = fmaxf(0.001f, sqr(mat->roughness) / aspect);                    //ax随着参数anisotropic的增加而增加
		float ay = fmaxf(0.001f, sqr(mat->roughness) * aspect);                    //ay随着参数anisotropic的增加而减少，ax和ay在anisotropic值为0时相等
		float Ds = GTR2_aniso(NdotH, dot(H, X), dot(H, Y), ax, ay);  //各项异性GTR2导出对应H的法线强度
		float FH = SchlickFresnel(LdotH);  //返回菲尼尔核心，既pow(1 - cosθd, 5)
		float3 Fs = lerp(Cspec0, make_float3(1.0f), FH); //菲尼尔项，使用了Cspec0作为F0，既高光底色，模拟金属的菲涅尔染色
		float Gs;   //几何项，一般与l，v和n相关，各项异性时还需要考虑方向空间中的切线t和副切线b
		Gs = smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay);  //遮蔽关联的几何项G1
		Gs *= smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay); //阴影关联的几何项G1，随后合并两项存入Gs

		// sheen 光泽项，本身作为边缘处漫反射的补偿
		float3 Fsheen = FH * mat->sheen * Csheen; //迪士尼认为sheen值正比于菲涅尔项FH，同时强度被控制变量sheen和颜色控制变量Csheen影响

		// clearcoat(ior = 1.5->F0 = 0.04)
		//清漆层没有漫反射，只有镜面反射，使用独立的D, F和G项
		//下面行使用GTR1（berry）分布函数获取法线强度，第二个参数是a（粗糙度）
		//迪士尼使用用户控制变量clearcoatGloss，在0.1到0.001线性插值获取a
		float Dr = GTR1(NdotH, lerp(0.1f, 0.001f, mat->clearcoatGloss));
		float Fr = lerp(0.04f, 1.0f, FH); //菲涅尔项上调最低值至0.04
		float Gr = smithG_GGX(NdotL, 0.25f) * smithG_GGX(NdotV, .25f);   //几何项使用各项同性的smithG_GGX计算，a固定给0.25

		//（漫反射 + 光泽） * 非金属度 + 镜面反射 + 清漆高光
		// 注意漫反射计算时使用了subsurface控制变量对基于菲涅尔的漫反射 和 次表面散射进行插值过渡；此外还补上了之前提到的baseColor / pi
		// 使用非金属度（既：1 - 金属度）用以消除来自金属的漫反射 < -非物理，但是能插值很爽啊
		return ((1.0f / M_PIf) * lerp(Fd, ss, mat->subsurface) * Cdlin + Fsheen)
			* (1.0f - mat->metallic)
			+ Gs * Fs * Ds + 0.25f * mat->clearcoat * Gr * Fr * Dr;
	}

	__forceinline__ __device__ float bxdfPdf(const Material::intersection& its, const float3& wo, const float3& wi, Material::disney_material* mat)
	{
			// 分别计算三种 BRDF 的概率密度
		float pdf_diffuse = DisneyDiffusePdf(its, wo,wi,mat );
		float pdf_specular = DisneySpecularPdf(its, wo, wi, mat);
		float pdf_clearcoat = DisneyClearcoatPdf(its, wo, wi, mat);
		// 辐射度统计
		float r_diffuse = (1.0 - mat->metallic);
		float r_specular = 1.0;
		float r_clearcoat = 0.25 * mat->clearcoat;
		float r_sum = r_diffuse + r_specular + r_clearcoat;

		// 根据辐射度计算选择某种采样方式的概率
		float p_diffuse = r_diffuse / r_sum;
		float p_specular = r_specular / r_sum;
		float p_clearcoat = r_clearcoat / r_sum;

		// 根据概率混合 pdf
		float pdf = p_diffuse * pdf_diffuse
			+ p_specular * pdf_specular
			+ p_clearcoat * pdf_clearcoat;

		pdf = fmaxf(1e-10f, pdf);
		return pdf;

	}

	__forceinline__ __device__ float3 bxdfSample(const Material::intersection& its, const float3& wo, float3& wi, float& pdf, Material::disney_material* mat)
	{
		float2 rand2 = make_float2(curand_uniform(its.state), curand_uniform(its.state2));
		// 辐射度统计
		float r_diffuse = (1.0 - mat->metallic);
		float r_specular = 1.0;
		float r_clearcoat = 0.25 * mat->clearcoat;
		float r_sum = r_diffuse + r_specular + r_clearcoat;

		// 根据辐射度计算概率
		float p_diffuse = r_diffuse / r_sum;
		float p_specular = r_specular / r_sum;
		float p_clearcoat = r_clearcoat / r_sum;

		float rd = curand_uniform(its.state);

		// 漫反射
		if (rd <= p_diffuse) {
			wi = sampleDisneyDiffuse(its,rand2,wo,wi,mat);
		}
		// 镜面反射
		else if (rd <= (p_diffuse + p_specular)) {
			wi = sampleDisneySpecular(its, rand2, wo, wi, mat);
		}
		// 清漆
		else {
			wi = sampleDisneyClearcoat(its, rand2, wo, wi, mat);
		}
		pdf = bxdfPdf(its,wo,wi,mat);
		return evalBxdf(its, wo, wi, mat);

	}
}
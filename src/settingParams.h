#pragma once

#include <vector>
#include <utility>
#include <string>
#include <tuple>

struct Float3;

enum class MiePhaseFunctionType
{
	HenyeyGreenstein,
	Mie,
};

enum class UiWidgetType
{
	Scalar, Input, Checkbox,
};

enum class ToneMappingType
{
	Uncharted, ACES1, ACES2, Reinhard,
};

struct SkyParams
{
	std::vector<std::tuple<float*, std::string, UiWidgetType, float, float, bool>> GetValueList()
	{
		return {
			{ &timeOfDay      , "Time of Day"           , UiWidgetType::Scalar, 0.01f , 0.99f , false },
			{ &sunAxisAngle   , "Sun Axis Angle"        , UiWidgetType::Scalar, 5.0f , 85.0f, false },
			{ &skyScalar      , "Sky Scalar"            , UiWidgetType::Input , 0.01f, 1.0f , false },
			{ &sunScalar      , "Sun Scalar"            , UiWidgetType::Input , 0.01f, 1.0f , false },
			{ &sunAngle       , "Sun Angle"             , UiWidgetType::Input , 0.01f, 1.0f , false },
		};
	}

	bool needRegenerate = true;

	float timeOfDay      = 0.25f;
	float sunAxisAngle   = 45.0f;
	float skyScalar      = 0.01f;
	float sunScalar      = 0.01f;
	float sunAngle       = 5.0f;
};

struct SampleParams
{
	std::vector<std::tuple<bool*, float*, std::string, UiWidgetType, float, float, bool>> GetValueList()
	{
		return {
			{ &sampleSurfaceVsLightUseMisWeight, nullptr , "Surface vs Light Use MIS Weight" , UiWidgetType::Checkbox, 0.0f , 1.0f , false },
			{ &sampleSkyVsSunUseFluxWeight, nullptr , "Sky vs Sun Use Flux Weight" , UiWidgetType::Checkbox, 0.0f , 1.0f , false },
			{ nullptr, &sampleSurfaceVsLight , "Surface vs Light Sample Probablity" , UiWidgetType::Scalar, 0.0f , 1.0f , false },
			{ nullptr, &sampleSkyVsSun , "Sky vs Sun Sample Probablity" , UiWidgetType::Scalar, 0.0f , 1.0f , false },
		};
	}

	bool sampleSurfaceVsLightUseMisWeight = true;
	bool sampleSkyVsSunUseFluxWeight = true;
	float sampleSurfaceVsLight = 0.5f;
	float sampleSkyVsSun = 0.5f;
};

struct RenderPassSettings
{
	std::vector<std::pair<bool*, std::string>> GetValueList()
	{
		return {
			{ &enableTemporalDenoising  , "Enable Temporal Denoising"   },
			{ &enableLocalSpatialFilter , "Enable Local SpatialFilter " },
			{ &enableNoiseLevelVisualize, "Enable Noise Level Visualize"},
			{ &enableWideSpatialFilter  , "Enable Wide Spatial Filter"  },
			{ &enableTemporalDenoising2 , "Enable Temporal Denoising 2" },
			{ &enablePostProcess        , "Enable Post Process"         },
			{ &enableDownScalePasses    , "Enable Down Scale Passes"    },
			{ &enableHistogram          , "Enable Histogram"            },
			{ &enableAutoExposure       , "Enable Auto Exposure"        },
			{ &enableBloomEffect        , "Enable Bloom Effect"         },
			{ &enableLensFlare          , "Enable Lens Flare"           },
			{ &enableToneMapping        , "Enable Tone Mapping"         },
			{ &enableSharpening         , "Enable Sharpening"           }
		};
	}

	bool enableTemporalDenoising   = true;
	bool enableLocalSpatialFilter  = true;
	bool enableNoiseLevelVisualize = false;
	bool enableWideSpatialFilter   = true;
	bool enableTemporalDenoising2  = true;
	bool enablePostProcess         = true;
	bool enableDownScalePasses     = true;
	bool enableHistogram           = true;
	bool enableAutoExposure        = true;
	bool enableBloomEffect         = false;
	bool enableLensFlare           = false;
	bool enableToneMapping         = true;
	bool enableSharpening          = false;
};

struct PostProcessParams
{
	std::vector<std::tuple<float*, std::string, float, float, bool>> GetValueList()
	{
		return {
			{ &exposure, "Exposure", 0.01f, 100.0f, true},

			{ &A, "Tonemap A", 0.0f,  1.0f  , false},
			{ &B, "Tonemap B", 0.0f,  1.0f  , false},
			{ &C, "Tonemap C", 0.0f,  1.0f  , false},
			{ &D, "Tonemap D", 0.0f,  1.0f  , false},
			{ &E, "Tonemap E", 0.0f,  1.0f  , false},
			{ &F, "Tonemap F", 0.0f,  1.0f  , false},
			{ &W, "Tonemap W", 1.0f, 1e10f, true },

			{ &gamma, "Gamma", 1.0f, 5.0f, false }
		};
	}

	ToneMappingType toneMappingType = ToneMappingType::Reinhard;

	float exposure = 1.0f;

	float A = 0.15f;
	float B = 0.50f;
	float C = 0.10f;
	float D = 0.20f;
	float E = 0.02f;
	float F = 0.30f;
	float W = 10.0f;

	float gamma = 2.2f;
};

struct DenoisingParams
{
	std::vector<std::pair<float*, std::string>> GetValueList()
	{
		return {
			{ &sigma_normal  , "sigma_normal" },
			{ &sigma_depth   , "sigma_depth" },
			{ &sigma_material, "sigma_material" }
		};
	}

	float sigma_normal   = 100.0f;
	float sigma_depth    = 0.1f;
	float sigma_material = 100.0f;
};
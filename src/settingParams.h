#pragma once

#include <vector>
#include <utility>
#include <string>

struct Float3;

enum class MiePhaseFunctionType
{
	HenyeyGreenstein,
	Mie,
};

struct SkyParams
{
	// Observer
	float latitude = 37.774929f;
	float longtitude = -122.419418f;
	float elevation = 16.0f;
	int month = 1;
	int day = 25;
	float timeOfDay = 0.22f;

	// Quality
	int numSamples = 32;
	int numLightRaySamples = 16;

	// Sun
	float sunPower = 20.0f;

	// Earth
	float earthRadius = 6360.0f;
	float atmosphereHeight = 60.0f;

	// Rayleigh
	Float3 mfpKmRayleigh = Float3(181.818f, 76.923f, 44.642f);
	float atmosphereThickness = 8000.0f;

	// Mie
	Float3 mfpKmMie = Float3(43.478f);
	Float3 albedoMie = Float3(0.91f);
	float g = 0.76;
	float aerosolThickness = 1200.0f;
	MiePhaseFunctionType miePhaseFuncType = MiePhaseFunctionType::Mie;

	// other
	int customData = 0;
};

struct RenderPassSettings
{
	std::vector<std::pair<bool*, std::string>> GetValueList()
	{
		return {
			{ &enableDenoiseReconstruct , "Enable Denoise Reconstruct"  },
			{ &enableTemporalDenoising  , "Enable Temporal Denoising"   },
			{ &enableLocalSpatialFilter , "Enable Local SpatialFilter " },
			{ &enableNoiseLevelVisualize, "Enable Noise Level Visualize"},
			{ &enableWideSpatialFilter  , "Enable Wide Spatial Filter"  },
			{ &enableTemporalDenoising2 , "Enable Temporal Denoising 2" },
			{ &enablePostProcess        , "Enable Post Process"         },
			{ &enableBloomEffect        , "Enable Bloom Effect"         },
			{ &enableLensFlare          , "Enable Lens Flare"           },
			{ &enableToneMapping        , "Enable Tone Mapping"         },
			{ &enableSharpening         , "Enable Sharpening"           }
		};
	}

	bool enableDenoiseReconstruct  = false;
	bool enableTemporalDenoising   = true;
	bool enableLocalSpatialFilter  = true;
	bool enableNoiseLevelVisualize = false;
	bool enableWideSpatialFilter   = true;
	bool enableTemporalDenoising2  = true;
	bool enablePostProcess         = true;
	bool enableBloomEffect         = false;
	bool enableLensFlare           = false;
	bool enableToneMapping         = true;
	bool enableSharpening          = false;
};
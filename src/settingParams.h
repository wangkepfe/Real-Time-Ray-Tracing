#pragma once

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
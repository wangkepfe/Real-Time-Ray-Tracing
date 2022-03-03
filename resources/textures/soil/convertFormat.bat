magick albedo.tif -channel RGB -separate tmp/albedo_sep.png
magick tmp/albedo_sep-0.png tmp/albedo_sep-1.png tmp/albedo_sep-2.png roughness.tif -channel RGBA -combine out/albedoRoughness.png

magick normal.tif -channel RGB -separate tmp/normal_sep.png
magick tmp/normal_sep-0.png tmp/normal_sep-1.png tmp/normal_sep-2.png height.tif -channel RGBA -combine out/normalHeight.png

@REM Verify
@REM magick out/albedoRoughness.png -channel RGBA -separate tmp/albedoRoughness_sep.png
@REM magick out/normalHeight.png -channel RGBA -separate tmp/normalHeight_sep.png
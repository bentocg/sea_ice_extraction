for PATCH_SIZE in 300 500 700 900 1200
do
	python sea_ice_extraction_pipeline.py --input_dir /home/bento/imagery --output_shp test_output/$PATCH_SIZE/sea_ice.shp --patch_size=$PATCH_SIZE
done

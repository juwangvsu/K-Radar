# radar_zyx_cube/cube_00621.mat  
	raw pw measurement, 
	unlogged, -1 for out of fov, 
	otherwise range from single digit to 1e12, dynamic range about 50db
# radar_tesseract/tesseract_00621.mat 
	raw pw measurement, unloged
# radar_bev_image/radar_bev_100_00621.png

# processed:
# RadarTensor/rdr_polar_3d/polar3d_00621.npy 
	stock max 74.06, min 0.00033, most likely normalized but not logged since the min value is not negative. if logged, will probably see some negative db value. 
# RadarTensor/rdr_polar_3d/new_all/1/polar3d_00621.npy local gen,  

datasets/kradar_detection_v2_1.py
	4d to rdr_polar_3d
	kradar_detection.save_polar_3d

task:
	convert zyx to bev and verify if result is consistent with cthe rdr

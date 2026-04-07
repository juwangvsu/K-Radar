

--------------- 4/6/26 more readme  ------------------

see seg_deeplabv3/readme_kradar.txt

--------------- 3/31/26 code repo ------------------
alien3
	~/Documents/K-Radar
gpuhead
	/data/jwang/Documents/K-Radar

--------------- 3/31/26 datasets repo ------------------
alien3
	~/Documents/datasets/k-radar
gpuhead
	/data/jwang/datasets/k-radar


cam crop front:
	python3 crop_left.py ../datasets/k-radar/20/

----------------------------------------
# radar_zyx_cube/cube_00621.mat  
	raw pw measurement, mean over all dopplor channel
	unlogged, -1 for out of fov, 
	otherwise range from single digit to 1e12, dynamic range about 50db
 	index 70 crosspond to 0-meter height. z[0] correspond to -30 degree in eleva
tion. this also explain the blackout area of the generated bev img.

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

----------dataset repo ---------------------
https://github.com/juwangvsu/K-Radar/blob/main/docs/dataset.md
https://drive.google.com/drive/folders/1IfKu-jKB1InBXmfacjMKQ4qTm8jiHrG_
https://kaistavelab.tw5.quickconnect.to/
	 kradards Password : Kradar2022

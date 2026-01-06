#------------1/6/26 .mat to .png--------------------------------
# ~/Documents/datasets/k-radar/1/radar_tesseract/*.mat 4d radar tensor
#~/Documents/datasets/k-radar/RadarTensor/rdr_polar_3d/new_all/1/*.npy 3d rae tensor
# *.png: range-angle 2d image, elevation axis summed 
alien3
conda activate kradar
cd Documents/K-Radar
export DISPLAY=:1
python datasets/kradar_detection_v2_1.py 
python radar_rdr_polar.py --polar_file ~/Documents/datasets/k-radar/RadarTensor/rdr_polar_3d/new_all/1
cp *.png /tmp



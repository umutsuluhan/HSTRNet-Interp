main:
	python3 test_video.py --videos videos/car_vid.mp4 videos/car_vid_2X_12fps.mp4
train: 
	python3 train.py --data_root /home/ortak/mughees/datasets/vimeo_triplet
clean:
	rm -rf __pycache__ ./model/__pycache_ project.mp4 logs/* model_dict/*

main:
	python3 test_video.py
train: 
	python3 train.py --data_root /home/hus/Desktop/data/vimeo_triplet
clean:
	rm -rf __pycache__ ./model/__pycache_ project.mp4 logs

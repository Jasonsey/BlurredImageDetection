all:
	cd dataset && python create_dataset.py
	echo 'Dataset Created~!'
	python train.py
	echo 'Trained ~!'
	python detection.py
	echo 'Detected ~!'

clean_model:
	rm -r ../../../data/output/total_image/models

clean_output:
	rm -r ../../../data/output/total_image/output

clean_log:
	rm -r ../../../data/output/total_image/log

clean:
	rm -r ../../../data/output/total_image
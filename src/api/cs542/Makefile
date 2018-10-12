all:
	python dataset.py
	echo 'Dataset Created~!'
	python train.py
	echo 'Trained ~!'
	python detection.py
	echo 'Detected ~!'

clean_model:
	rm -r ../../../data/output/cs542/models

clean_output:
	rm -r ../../../data/output/cs542/output

clean_log:
	rm -r ../../../data/output/cs542/log

clean:
	rm -r ../../../data/output/cs542
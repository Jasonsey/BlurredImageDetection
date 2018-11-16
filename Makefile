.PHONY: server client thrift pip_dependency deploy

server:
	cd src && python3.6 main.py server

client:
	cd src && python3.6 main.py client

thrift:
	mkdir -p src/api/thrift_api && thrift -r -gen py -out thrift_api interface.thrift

pip_dependency: 
	pip install --no-cache-dir \
		--trusted-host \
		mirrors.aliyun.com \
		-i http://mirrors.aliyun.com/pypi/simple/ \
		-q -r requirements.txt

deploy: pip_dependency thrift server

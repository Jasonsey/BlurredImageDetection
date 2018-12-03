version=1.2.0

name=blur_detection
base_name=${name}_base

remote=172.18.31.204:3001/linjx/${name}:${version}
remote_base=172.18.31.204:3001/linjx/${base_name}

.PHONY: server client thrift build_base_image rm_base_image rebuild_base_image up down tag update


server:
	cd src && python3.6 main.py server
client:
	cd src && python3.6 main.py client
thrift:
	mkdir -p src/api/thrift_api && thrift -r -gen py -out thrift_api interface.thrift


build_base_image:
	docker build -f Dockerfile.base -t ${base_name} .
	docker tag ${base_name} ${remote_base}
	docker push ${remote_base}
rm_base_image:
	docker rmi ${base_name} ${remote_base}
rebuild_base_image: rm_base_image build_base_image


up:
	docker-compose up -d
down:
	docker-compose down --rmi all
tag:
	docker tag ${name} ${remote}
	docker push ${remote}
update: down up tag

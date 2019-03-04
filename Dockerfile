FROM 172.18.31.204:3001/linjx/blur_detection_base

LABEL Name=blurred_image_detection Version=1.2.1
EXPOSE 9099

ENV APP_ROOT /app
WORKDIR ${APP_ROOT}

ADD . ${APP_ROOT}

RUN python3.6 -O -m compileall src && \
	find . -name "*.pyc" -exec rename "s/.cpython-36.opt-1//" {} \; && \
	find . -name '*.pyc' -execdir mv {} .. \; && \
	find . -name '*.py' -type f -print -exec rm {} \;

# Up Server
CMD ["make", "server"]

# For Deploy
# WORKDIR ${APP_ROOT}/src
# ENTRYPOINT ["python3.6", "main.py"]
# CMD ["server", "--gpu", "2"]

FROM 172.18.31.204:3001/linjx/blur_detection_base

LABEL Name=blurred_image_detection Version=1.2.0
EXPOSE 9099

ENV APP_ROOT /app
WORKDIR ${APP_ROOT}

ADD . ${APP_ROOT}

# Up Server
CMD ["make", "server"]

# For Deploy
# WORKDIR ${APP_ROOT}/src
# ENTRYPOINT ["python3.6", "main.py"]
# CMD ["server", "--gpu", "2"]

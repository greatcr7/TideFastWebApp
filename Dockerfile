FROM python:3.12

ENV TZ=Asia/Shanghai
RUN apt-get update && apt-get install -y tzdata

WORKDIR app/

COPY ./app .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "tidefast.py"]

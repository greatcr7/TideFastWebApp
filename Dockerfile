FROM python:1-3.12-bullseye

WORKDIR /

# COPY ./app .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

EXPOSE 8501
CMD ["streamlit", "run", "市场.py"]
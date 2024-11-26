FROM python:3.10-slim
WORKDIR /usr/src/DataAnalysis
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . ./
ENV PORT 8501
EXPOSE $PORT
CMD streamlit run app/main.py
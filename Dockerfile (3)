FROM python:3.10-slim
WORKDIR /app
RUN pip install flask
COPY portal.html .
COPY app.py .
EXPOSE 7860
CMD ["python", "app.py"]

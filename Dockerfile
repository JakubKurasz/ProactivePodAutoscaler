# Pull in Python build of CPA
FROM custompodautoscaler/python:latest

ADD             ./requirements.txt ./
ADD             ./model.pkl ./
ADD             ./sc.joblib ./
RUN             python3 -m pip install -r requirements.txt
# Add config, evaluator and metric gathering Py scripts
ADD config.yaml evaluate.py metric.py /
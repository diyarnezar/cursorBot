FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install streamlit shap lightgbm xgboost stable-baselines3 optuna ray[default] plotly
EXPOSE 8501
ENTRYPOINT ["/bin/bash"]
# To run the dashboard: streamlit run modules/dashboard.py
# To run the bot: python ultra_main.py 
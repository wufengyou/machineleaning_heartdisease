FROM python:3.11.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# Install pipenv
RUN pip install pipenv
COPY Pipfile* ./

RUN pipenv install --deploy --system

# Copy requirements first for better caching
# COPY requirements.txt .

# Install dependencies
# RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run the application
ENTRYPOINT ["streamlit", "run", "app1.py", "--server.address", "0.0.0.0"]

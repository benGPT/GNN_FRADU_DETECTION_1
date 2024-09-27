
# Base image
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy all files to working directory
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port for Flask app
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]

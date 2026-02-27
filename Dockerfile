FROM python:3.8-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p models logs data/cs-train

# Expose port
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"]

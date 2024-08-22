# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies (optional but recommended for production)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create and set permissions for the database directory
RUN mkdir -p /app/db && \
    chmod 775 /app/db && \
    chown -R www-data:www-data /app/db

# Copy the current directory contents into the container at /app
COPY . /app/

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port 7860 (according to Hugging Face)
EXPOSE 7860

# Run the Django development server on port 7860
CMD ["python", "manage.py", "runserver", "0.0.0.0:7860"]

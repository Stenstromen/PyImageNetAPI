# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Set environment variables
ENV AUTHORIZATION_KEY "123"
ENV CORS_ORIGINS "*"

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn
RUN pip install --no-cache-dir gunicorn

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the command to start gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:80", "app:app"]

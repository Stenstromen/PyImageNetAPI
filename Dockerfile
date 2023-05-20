# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add contents of the current directory (on your machine) to the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN mkdir -p ~/.keras/models && \
pip install --no-cache-dir -r requirements.txt 

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]

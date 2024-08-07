# Use an official Python runtime as a parent image
FROM python:3.12.2

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./backend /app 

# Ensure the correct permissions are set
RUN chmod -R 755 /app

# Exclude .env file
RUN rm -f /app/.env

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run main.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
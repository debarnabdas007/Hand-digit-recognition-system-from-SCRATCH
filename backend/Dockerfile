#This says, "Go to the internet, download a tiny, blank Linux computer, and install Python 3.10 on it."
FROM python:3.10-slim   

# Set the working directory inside the container :- Create a folder inside this mini-computer called /app and do all the following steps inside it.
WORKDIR /app

# Copy the requirements file first (this caches the heavy PyTorch download) :- Copy my local requirements file and put it in the mini-computer.
COPY requirements.txt .

# Install dependencies  # Run the installation. This keeps the massive PyTorch download completely trapped inside the mini-computer."
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code into the container   :- Copy the rest of my backend code (the API, the PyTorch model) into the mini-computer.
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to ignite the server when the container starts   :- When the mini-computer turns on, run this specific terminal command to start the server.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
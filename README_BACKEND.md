# VLM Realtime Webcam Application

A real-time webcam application that uses Vision Language Models (VLM) to analyze live camera feed. The application consists of an HTML frontend and a Python backend that communicates with a llama.cpp VLM server.

## Architecture

```
HTML Frontend (index.html) 
    ↓ HTTP requests
Python Backend (backend.py) 
    ↓ Proxy requests
llama.cpp VLM Server
```

## Features

- Real-time webcam capture and analysis
- Configurable request intervals (100ms to 2s)
- Image preprocessing and optimization
- Error handling and connection monitoring
- Health check endpoints
- CORS support for cross-origin requests

## Setup Instructions

### Prerequisites

1. **Python 3.8+** installed on your system
2. **llama.cpp server** running with a VLM model (like SmolVLM)
3. A web browser that supports webcam access
4. HTTPS or localhost environment (required for webcam access)

### Step 1: Set up the llama.cpp VLM Server

1. Download and compile llama.cpp with VLM support
2. Download a VLM model (e.g., SmolVLM)
3. Start the llama.cpp server:
   ```bash
   ./server -m path/to/your/vlm-model.gguf --port 8080 --host localhost
   ```
   
   Make sure the server is running on `http://localhost:8080` (default) or note the URL for configuration.

### Step 2: Set up the Python Backend

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Backend Server**:
   
   **Option A: Using the batch file (Windows)**:
   ```bash
   start_backend.bat
   ```
   
   **Option B: Manual startup**:
   ```bash
   python backend.py --host localhost --port 5000 --vlm-url http://localhost:8080
   ```

### Step 3: Open the Frontend

1. Open `index.html` in a web browser
2. Grant webcam permissions when prompted
3. The default API URL should be `http://localhost:5000` (pointing to the Python backend)

## Configuration

### Backend Configuration

The backend can be configured via command line arguments:

```bash
python backend.py --help
```

Options:
- `--host`: Host to bind to (default: localhost)
- `--port`: Port to bind to (default: 5000)
- `--vlm-url`: VLM server URL (default: http://localhost:8080)
- `--debug`: Enable debug mode

### Frontend Configuration

- **Base API URL**: Change in the frontend interface or modify the default in `index.html`
- **Request Interval**: Configurable via dropdown (100ms to 2s)
- **Instruction**: Customize the prompt sent to the VLM

## API Endpoints

### Backend Endpoints

- `POST /v1/chat/completions`: Main endpoint that proxies requests to VLM server
- `GET /health`: Health check for both backend and VLM server
- `GET /config`: Get current configuration
- `POST /config`: Update VLM server URL

### Health Check Example

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "vlm_server": "healthy",
  "vlm_url": "http://localhost:8080"
}
```

## Usage

1. **Start the llama.cpp VLM server** (usually on port 8080)
2. **Start the Python backend** (on port 5000)
3. **Open the frontend** in a web browser
4. **Grant webcam permissions**
5. **Enter your instruction** (e.g., "What do you see?", "Describe the scene", "Count the objects")
6. **Click Start** to begin real-time analysis
7. **View responses** in the response textarea

## Troubleshooting

### Common Issues

1. **Camera Access Denied**:
   - Ensure you're accessing via HTTPS or localhost
   - Check browser permissions for camera access
   - Try refreshing the page and granting permissions again

2. **Backend Connection Error**:
   - Verify the Python backend is running on port 5000
   - Check if the Base API URL in frontend matches the backend URL
   - Look for CORS errors in browser console

3. **VLM Server Connection Error**:
   - Ensure llama.cpp server is running and accessible
   - Verify the VLM server URL in backend configuration
   - Check the `/health` endpoint for server status

4. **Slow Response Times**:
   - Increase the request interval
   - Check if the VLM model is appropriate for real-time inference
   - Monitor CPU/GPU usage

### Debug Mode

Enable debug mode for more verbose logging:

```bash
python backend.py --debug
```

## File Structure

```
smolvlm-realtime-webcam-main/
├── index.html              # Frontend interface
├── backend.py              # Python backend server
├── requirements.txt        # Python dependencies
├── start_backend.bat       # Windows startup script
└── README.md              # This file
```

## Dependencies

### Python Backend
- Flask 2.3.3: Web framework
- flask-cors 4.0.0: CORS support
- requests 2.31.0: HTTP client
- Pillow 10.0.1: Image processing

### Frontend
- Pure HTML/CSS/JavaScript (no external dependencies)
- Modern browser with webcam support

## Performance Optimization

The backend includes several optimizations:

1. **Image Compression**: Automatic JPEG compression with 85% quality
2. **Image Resizing**: Automatic resizing to max 512x512 pixels
3. **Request Timeouts**: 30-second timeout for VLM requests
4. **Error Handling**: Graceful error handling and user feedback

## Security Considerations

- The application is designed for local use only
- No authentication or authorization mechanisms included
- Camera data is processed locally and sent to specified VLM server only
- Consider security implications if deploying on public networks

## License

See LICENSE file for details.

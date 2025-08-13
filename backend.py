#!/usr/bin/env python3
"""
Python backend server for VLM webcam application.
Acts as a proxy between the HTML frontend and llama.cpp VLM server.
"""

import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

# Try to import configuration, fallback to defaults
try:
    from config import (
        VLM_SERVER_URL, BACKEND_HOST, BACKEND_PORT,
        REQUEST_TIMEOUT, LOG_LEVEL
    )
    LLAMA_CPP_SERVER_URL = VLM_SERVER_URL
except ImportError:
    # Fallback to defaults if config.py is not available
    LLAMA_CPP_SERVER_URL = "http://localhost:8080"
    REQUEST_TIMEOUT = 30
    LOG_LEVEL = "INFO"

# Configure logging
log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    Proxy endpoint that mimics OpenAI's chat completions API.
    Forwards requests to llama.cpp server.
    """
    try:
        # Get request data from frontend
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        logger.info("Received chat completion request")
        
        # Log that we received an image (no processing needed)
        messages = data.get('messages', [])
        has_image = False
        for message in messages:
            if isinstance(message.get('content'), list):
                for content_item in message['content']:
                    if content_item.get('type') == 'image_url':
                        has_image = True
                        break
        
        if has_image:
            logger.info("Forwarding image directly to VLM (no processing)")
        
        # Forward request to llama.cpp server
        try:
            response = requests.post(
                f"{LLAMA_CPP_SERVER_URL}/v1/chat/completions",
                json=data,
                headers={'Content-Type': 'application/json'},
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                response_data = response.json()
                logger.info("Successfully received response from VLM")
                return jsonify(response_data)
            else:
                logger.error(f"VLM server error: {response.status_code}")
                return jsonify({
                    "error": f"VLM server error: {response.status_code}",
                    "detail": response.text
                }), response.status_code
                
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to VLM server")
            return jsonify({
                "error": "Cannot connect to VLM server",
                "detail": f"Make sure llama.cpp server is running on {LLAMA_CPP_SERVER_URL}"
            }), 503
            
        except requests.exceptions.Timeout:
            logger.error("VLM server timeout")
            return jsonify({
                "error": "VLM server timeout",
                "detail": "The VLM server took too long to respond"
            }), 504
            
        except Exception as e:
            logger.error(f"Error communicating with VLM server: {e}")
            return jsonify({
                "error": "VLM server communication error",
                "detail": str(e)
            }), 500
    
    except Exception as e:
        logger.error(f"Backend error: {e}")
        return jsonify({
            "error": "Backend processing error",
            "detail": str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check if VLM server is reachable
        response = requests.get(f"{LLAMA_CPP_SERVER_URL}/health", timeout=5)
        vlm_status = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        vlm_status = "unreachable"
    
    return jsonify({
        "status": "healthy",
        "vlm_server": vlm_status,
        "vlm_url": LLAMA_CPP_SERVER_URL
    })


@app.route('/config', methods=['GET', 'POST'])
def config():
    """
    Configuration endpoint to get/set VLM server URL.
    """
    global LLAMA_CPP_SERVER_URL
    
    if request.method == 'GET':
        return jsonify({
            "vlm_server_url": LLAMA_CPP_SERVER_URL,
            "note": "Images are passed directly to VLM without processing"
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        if data and 'vlm_server_url' in data:
            LLAMA_CPP_SERVER_URL = data['vlm_server_url'].rstrip('/')
            logger.info(f"Updated VLM server URL to: {LLAMA_CPP_SERVER_URL}")
            return jsonify({"message": "Configuration updated", "vlm_server_url": LLAMA_CPP_SERVER_URL})
        else:
            return jsonify({"error": "Invalid configuration data"}), 400


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='VLM Backend Server')
    parser.add_argument('--host', default=BACKEND_HOST if 'BACKEND_HOST' in globals() else 'localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=BACKEND_PORT if 'BACKEND_PORT' in globals() else 5000, help='Port to bind to')
    parser.add_argument('--vlm-url', default=LLAMA_CPP_SERVER_URL, help='VLM server URL')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    LLAMA_CPP_SERVER_URL = args.vlm_url.rstrip('/')
    
    logger.info(f"Starting VLM backend server on {args.host}:{args.port}")
    logger.info(f"VLM server URL: {LLAMA_CPP_SERVER_URL}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

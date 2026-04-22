"""
Start both FastAPI backend + React frontend with one command.
Usage:  cd client/ && python start.py
"""

import subprocess
import sys
import os
import time
import signal

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(base_dir, "frontend")

    # Check if node_modules exists
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        print("📦 Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
        print("✅ Dependencies installed\n")

    # Check if model exists
    model_path = os.path.join(base_dir, "..", "models", "nsfw_detector.onnx")
    if not os.path.exists(model_path):
        print("⚠️  WARNING: models/nsfw_detector.onnx not found!")
        print("   Run export_onnx.py first to create the ONNX model.")
        print("   Starting anyway (backend will return errors until model exists).\n")

    print("🚀 Starting NSFW Detector...")
    print("   Frontend: http://localhost:3000")
    print("   Backend:  http://localhost:8000")
    print("   Press Ctrl+C to stop both\n")

    # Start backend
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=base_dir,
    )

    # Start frontend
    frontend = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
    )

    # Handle Ctrl+C gracefully
    def shutdown(sig, frame):
        print("\n\n⏹️  Shutting down...")
        backend.terminate()
        frontend.terminate()
        backend.wait(timeout=5)
        frontend.wait(timeout=5)
        print("✅ Stopped. Bye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait for either process to exit
    try:
        while True:
            if backend.poll() is not None:
                print(f"❌ Backend exited with code {backend.returncode}")
                break
            if frontend.poll() is not None:
                print(f"❌ Frontend exited with code {frontend.returncode}")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()

from flask import Flask, render_template, Response
import cv2
import numpy as np
from camera_class import VideoStreamer

app = Flask(__name__)
video_streamer = VideoStreamer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_streamer.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)

import cv2
import os
from rembg import remove
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile
from time import sleep

def get_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_rate

def limit_video_duration(input_video_path, max_duration=10):
    cap = cv2.VideoCapture(input_video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / frame_rate

    if duration > max_duration:
        temp_file = NamedTemporaryFile(delete=False, suffix='.mp4')
        out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        for _ in range(int(max_duration * frame_rate)):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        cap.release()
        return temp_file.name
    cap.release()
    return input_video_path

def split_video_to_frames(video_path, output_dir, progress_callback):
    cap = cv2.VideoCapture(video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")

        # Add brand name to the center of each frame
        height, width, _ = frame.shape
        cv2.putText(frame, "BGRM", (width // 2 - 50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
        progress_callback(frame_count / total_frames)
    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}")

def remove_background_from_frames(input_dir, output_dir, progress_callback):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = os.listdir(input_dir)
    total_files = len(files)
    for idx, frame_name in enumerate(files):
        input_path = os.path.join(input_dir, frame_name)
        output_path = os.path.join(output_dir, frame_name)
        
        with open(input_path, 'rb') as f:
            input_image = f.read()
        
        output_image = remove(input_image)
        
        # Load the processed image as a numpy array
        processed_frame = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Replace the transparent background with green color
        if processed_frame.shape[2] == 4:  # Check if the image has an alpha channel
            alpha_channel = processed_frame[:, :, 3]
            rgb_channels = processed_frame[:, :, :3]
            green_background = np.zeros_like(rgb_channels, dtype=np.uint8)
            green_background[:] = [0, 255, 0]  # Green color
            
            mask = alpha_channel == 0
            processed_frame = np.where(mask[:, :, None], green_background, rgb_channels)
        
        # Save the frame with the green background
        cv2.imwrite(output_path, processed_frame)
        progress_callback((idx + 1) / total_files)
    print(f"Background removed and replaced with green for frames in {input_dir} and saved to {output_dir}")

def merge_frames_to_video(frames_dir, output_video_path, frame_rate, progress_callback):
    frames = sorted(os.listdir(frames_dir))
    frame_path = os.path.join(frames_dir, frames[0])
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    total_frames = len(frames)
    for idx, frame_name in enumerate(frames):
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        out.write(frame)
        progress_callback((idx + 1) / total_frames)

    out.release()
    print(f"Video created at {output_video_path}")

def remove_photo_background(photo_file):
    with open(photo_file, 'rb') as f:
        input_image = f.read()
    
    output_image = remove(input_image)
    
    # Save the processed photo to a temporary file
    temp_file = NamedTemporaryFile(delete=False, suffix='.png')
    with open(temp_file.name, 'wb') as f:
        f.write(output_image)
    return temp_file.name

# Streamlit app
st.title("Video and Photo Background Removal")

option = st.radio("Choose an option:", ("Remove video background", "Remove photo background"))

if option == "Remove video background":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if video_file:
        temp_file = NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(video_file.read())
        temp_video_path = temp_file.name

        limited_video_path = limit_video_duration(temp_video_path)
        frame_rate = get_frame_rate(limited_video_path)
        
        frames_output_dir = 'frames'
        bg_removed_dir = 'bg_removed_frames'
        output_video_path = 'output_video.mp4'

        progress_bar = st.progress(0)

        split_video_to_frames(limited_video_path, frames_output_dir, progress_callback=lambda p: progress_bar.progress(int(p * 33)))
        remove_background_from_frames(frames_output_dir, bg_removed_dir, progress_callback=lambda p: progress_bar.progress(33 + int(p * 33)))
        merge_frames_to_video(bg_removed_dir, output_video_path, frame_rate, progress_callback=lambda p: progress_bar.progress(66 + int(p * 34)))

        st.video(output_video_path)

        with open(output_video_path, "rb") as video_file:
            btn = st.download_button(
                label="Download Processed Video",
                data=video_file,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )

elif option == "Remove photo background":
    photo_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])
    
    if photo_file:
        temp_file = NamedTemporaryFile(delete=False, suffix='.png')
        temp_file.write(photo_file.read())
        
        output_photo_path = remove_photo_background(temp_file.name)
        st.image(output_photo_path, caption="Processed Image")

        with open(output_photo_path, "rb") as image_file:
            btn = st.download_button(
                label="Download Processed Image",
                data=image_file,
                file_name="processed_image.png",
                mime="image/png"
            )

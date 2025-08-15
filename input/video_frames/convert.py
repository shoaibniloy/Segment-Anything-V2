import cv2
import os

def images_to_video(image_folder, output_video_path, fps=30, frame_size=None):
    """
    Convert a sequence of images to a video.

    :param image_folder: Folder containing the image sequence.
    :param output_video_path: Output path for the video file.
    :param fps: Frames per second for the video.
    :param frame_size: Tuple specifying the width and height of the frames (width, height).
                       If None, the size of the first image will be used.
    """
    # Get list of images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Sort images by name

    if not images:
        raise ValueError("No images found in the specified folder.")

    # Read the first image to determine the frame size
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)

    if frame_size is None:
        frame_size = (first_image.shape[1], first_image.shape[0])  # (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)

        # Resize frame if necessary
        if frame_size is not None and (frame.shape[1], frame.shape[0]) != frame_size:
            frame = cv2.resize(frame, frame_size)

        video_writer.write(frame)

    video_writer.release()

if __name__ == "__main__":
    image_folder = './'  # Replace with the path to your image folder
    output_video_path = 'video.mp4'  # Replace with the desired output video path
    fps = 1  # Frames per second
    frame_size = None  # Set to (width, height) if you want to specify frame size

    images_to_video(image_folder, output_video_path, fps, frame_size)

import cv2


class VideoRecorder:
    def __init__(self, filename, fps=20, frame_size=(256, 256)):
        self.video = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size
        )

    def write_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 0)
        self.video.write(frame)

    def release(self):
        self.video.release()
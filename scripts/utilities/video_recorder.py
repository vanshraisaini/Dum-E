import cv2


class VideoRecorder:
    def __init__(self, filename, fps=20, frame_size=(256, 256)):
        self.video = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size
        )

    def write_frame(self, frame, language_instruction):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 0)

        # Overlay the language instruction as text on the frame
        frame = self._add_text(frame, language_instruction)

        self.video.write(frame)

    def _add_text(self, frame, text, font_scale=0.4, thickness=1,
                   color=(255, 255, 255), bg_color=(0, 0, 0)):
        frame = frame.copy()
        h, w = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 5
        line_height = 15

        # Wrap text so it doesn't overflow the frame width
        words = text.split(" ")
        lines = []
        current_line = ""
        for word in words:
            test_line = (current_line + " " + word).strip()
            (test_w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if test_w > w - 2 * margin and current_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        if current_line:
            lines.append(current_line)

        # Draw a semi-transparent background band for readability
        band_height = margin * 2 + line_height * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, band_height), bg_color, -1)
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Draw each line of text
        for i, line in enumerate(lines):
            y = margin + line_height * (i + 1) - 4
            cv2.putText(frame, line, (margin, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return frame

    def release(self):
        self.video.release()
import cv2
import os
import pandas as pd
import numpy as np
import cvzone
import xgboost as xgb
from ultralytics import YOLO
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import threading
import io

# Define the path to the video file
video_path = "thief.mp4"

def send_email(receiver_email, frame):
    try:
        # Set up the SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Update if using a different service
        server.starttls()  # Enable security
        server.login('freedomtech85@gmail.com', 'mmgo wvcn eedr nxfn')  # Update with your credentials

        # Create the email
        msg = MIMEMultipart()
        msg['From'] = 'fredomtech85@gmail.com'  # Update with your email
        msg['To'] = receiver_email
        msg['Subject'] = 'Suspicious Activity Detected'

        # Encode the frame as an image
        _, buffer = cv2.imencode('.jpg', frame)
        img_data = buffer.tobytes()  # Convert the buffer to bytes

        # Attach the image
        img = MIMEImage(img_data)
        msg.attach(img)

        # Send the email
        server.send_message(msg)
        print("Email sent successfully.")

    except Exception as e:
        print(f"Failed to send email: {e}")

    finally:
        try:
            server.quit()  # Terminate the SMTP session
        except:
            pass  # If server was not initialized, just pass

def detect_shoplifting(video_path):
    # Load YOLOv8 model
    model_yolo = YOLO('yolo11s-pose.pt')  # Update with your model path

    # Load the trained XGBoost model
    model = xgb.Booster()
    model.load_model('trained_model.json')  # Update with your model path

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total Frames: {total_frames}")

    frame_tot = 0
    count = 0
    email_threads = []  # To track all the email threads

    while cap.isOpened() and frame_tot < total_frames:
        success, frame = cap.read()
        if not success:
            print("Warning: Frame could not be read. Skipping.")
            break  # Stop the loop if no frame is read

        count += 1
        if count % 3 != 0:
            continue

        # Resize the frame
        frame = cv2.resize(frame, (1018, 600))

        # Run YOLOv8 on the frame
        results = model_yolo(frame, verbose=False)

        # Visualize the YOLO results on the frame
        annotated_frame = results[0].plot(boxes=False)

        for r in results:
            bound_box = r.boxes.xyxy  # Bounding box coordinates
            conf = r.boxes.conf.tolist()  # Confidence levels
            keypoints = r.keypoints.xyn.tolist()  # Keypoints for human pose

            print(f'Frame {frame_tot}: Detected {len(bound_box)} bounding boxes')

            for index, box in enumerate(bound_box):
                if conf[index] > 0.80:  # Threshold for confidence score
                    x1, y1, x2, y2 = box.tolist()

                    # Prepare data for XGBoost prediction
                    data = {}
                    for j in range(len(keypoints[index])):
                        data[f'x{j}'] = keypoints[index][j][0]
                        data[f'y{j}'] = keypoints[index][j][1]

                    # Convert the data to a DataFrame
                    df = pd.DataFrame(data, index=[0])

                    # Prepare data for XGBoost prediction
                    dmatrix = xgb.DMatrix(df)

                    # Make prediction using the XGBoost model
                    sus = model.predict(dmatrix)
                    binary_predictions = (sus > 0.5).astype(int)
                    print(f'Prediction: {binary_predictions}')

                    # Annotate the frame based on prediction (0 = Suspicious, 1 = Normal)
                    if binary_predictions == 0:  # Suspicious
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cvzone.putTextRect(annotated_frame, f"{'Suspicious'}", (int(x1), (int(y1))), 1, 1)

                        # Create a thread to send an email with the current frame
                        receiver_email = "truckersfan66@gmail.com"  # Update with the actual receiver email
                        email_thread = threading.Thread(target=send_email, args=(receiver_email, annotated_frame))
                        email_threads.append(email_thread)  # Track the thread
                        email_thread.start()  # Start the email thread

                    else:  # Normal
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cvzone.putTextRect(annotated_frame, f"{'Normal'}", (int(x1), (int(y1))), 1, 1)

        # Show the annotated frame in a window
        cv2.imshow('Frame', annotated_frame)

        # Press 'q' to stop the video early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_tot += 1  # Increment frame counter

    # Wait for all email threads to finish
    for thread in email_threads:
        thread.join()  # Ensure all emails are sent before closing

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Call the function with the video path
detect_shoplifting(video_path)

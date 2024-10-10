import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os

def send_email(receiver_email, image_path, body="Hello, this is a test email with an image!"):
    """Send an email with an image attachment using SMTP. The sender's email and password are predefined."""
    
    # Predefined sender email and password
    sender_email = ""
    sender_password = ""  # Use an app password if 2FA is enabled

    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Test Email from Python with Image"

    # Attach the email body
    msg.attach(MIMEText(body, 'plain'))
   # body is the actual text content you want to send.
   # 'plain' specifies that this content is plain text.
    try:
        # Open the image file and attach it to the email
        with open(image_path, 'rb') as img:#'rb' mode is used to specify that the file should be opened in read binary mode.
            image = MIMEImage(img.read())
            image.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            msg.attach(image)

        # Create an SMTP session
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()  # Use TLS
        server.login(sender_email, sender_password)  # Login to the email account
        text = msg.as_string()  # Convert the message to a string
        server.sendmail(sender_email, receiver_email, text)  # Send the email
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        server.quit()  # Terminate the SMTP session


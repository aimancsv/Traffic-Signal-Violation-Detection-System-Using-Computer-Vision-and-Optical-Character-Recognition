import cv2
from pytesseract import pytesseract

# Set the path to the Tesseract OCR executable
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set the dimensions of the input image
imgWidth = 640
imgHeight = 480

# Load the Haar cascade classifier for license plate detection
plateDetector = cv2.CascadeClassifier("resources/haarcascade_russian_plate_number.xml")

# Set the minimum area size for license plate detection
minAreaSize = 500

# Set the OCR configuration for text extraction from license plates
ocrConfig = ('-l eng --oem 3 --psm 7')

# Function to detect license plate and extract the plate number
def detect_number_plate(img):
    # Convert the input image to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates in the grayscale image using the Haar cascade classifier
    detectedPlates = plateDetector.detectMultiScale(grayImg, 1.1, 4)

    # Iterate over the detected license plates
    for (startX, startY, width, height) in detectedPlates:
        # Extract the region of interest (license plate) from the image
        plateRoi = img[startY:startY + height, startX:startX + width]

        # Perform OCR to extract text from the license plate region
        extractedText = pytesseract.image_to_string(plateRoi, config=ocrConfig)

        # Process the extracted text
        if extractedText is not None:
            # Remove non-alphanumeric characters and convert to uppercase
            extractedText = ''.join(c for c in extractedText if c.isalnum() and c.upper() == c).strip()

            # Check if the extracted text is a valid license plate number
            if len(extractedText) >= 4:
                # Draw a rectangle around the detected license plate
                cv2.rectangle(img, (startX, startY), (startX + width, startY + height), (255, 0, 0), 2)

                # Display the extracted license plate region
                cv2.imshow("Number Plate", plateRoi)

                # Return the extracted license plate number
                return extractedText

    # If no valid license plate is found, return None
    return None

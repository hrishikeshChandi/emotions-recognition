import torch
import cv2 as cv
import torchvision
from PIL import Image
from torchvision import transforms

har_cascade = cv.CascadeClassifier("haar_face.xml")

model = torch.load("cpu_model.pth", weights_only=False)
model.eval()

img_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

class_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

capture = cv.VideoCapture(0)


def predict_emotion(img):
    img = Image.fromarray(img)
    img = img_transform(img).unsqueeze(0)

    with torch.inference_mode():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        label = logits.argmax(dim=1)
        max_prob = probs.max().item()
        emotion = class_names[label]

        return emotion, max_prob * 100


while True:
    success, frame = capture.read()
    if not success:
        print("webcam failed")
        break

    faces = har_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=6)

    for x, y, w, h in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        face = frame[y : y + h, x : x + w]
        emotion, confidence = predict_emotion(face)
        cv.putText(
            frame,
            f"{emotion.title()}",
            (x + 20, y - 20),
            cv.FONT_HERSHEY_DUPLEX,
            0.8,
            color=(0, 255, 255),
            thickness=2,
        )

        print(f"{emotion.title()} -> {round(confidence, 2)}%")
        with open("results.txt", "w") as f:
            f.write(f"{emotion.title()} -> {round(confidence, 2)}%")

    cv.imshow("emotions recognition", frame)

    if cv.waitKey(20) & 0xFF == ord("q"):
        break

capture.release()
cv.destroyAllWindows()

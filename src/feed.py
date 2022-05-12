import cv2
import onnxruntime as ort
import numpy as np


def main():
    index_to_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                       'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', "Delete", "Nothing", "Space"]

    ort_session = ort.InferenceSession("model.onnx")
    input_name = ort_session.get_inputs()[0].name

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x = cv2.resize(frame, (48, 48))

        x = x.reshape(1, 48, 48, 1).astype(np.float32)
        y = ort_session.run(None, {input_name: x})[0]

        index = np.argmax(y, axis=1)
        letter = index_to_letter[int(index)]

        cv2.putText(frame, letter, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), thickness=4)
        cv2.imshow("SLR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

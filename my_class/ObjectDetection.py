import torch
import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from io import BytesIO,BufferedReader
import random
import threading
import time


class CountdownTimer:
    def __init__(self):
        self.count = 3
        self.started = False
        self.finished = False

    def start(self):
        self.count = 3  # Reset count when starting
        self.timer_thread = threading.Thread(target=self.run_timer)
        self.timer_thread.daemon = True  # Set the thread as daemon so it terminates when the main thread exits
        self.started = True
        self.finished = False
        
        if not self.timer_thread.is_alive():  # Check if the thread is alive before starting
            self.timer_thread.start()
        

    def run_timer(self):
        while self.count > 0:
            # print(self.count)
            time.sleep(1)
            self.count -= 1
        # print("Go!")
        self.finished = True


class ObjectDetection:
    def __init__(self, capture_index):
        # default parameters
        self.capture_index = capture_index

        # model information
        self.model = YOLO("models/best.pt")

        # visual information
        self.annotator = None
        self.start_time = 0
        self.end_time = 0

        # device information
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def predict(self, im0):
        results = self.model(im0,verbose=False)
        return results

    def display_fps(self, im0):
        self.end_time = time.time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10
        cv2.rectangle(im0, (20 - gap, 70 - text_size[1] - gap), (20 + text_size[0] + gap, 70 + gap), (255, 255, 255), -1)
        cv2.putText(im0, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)


    def display_info(self,frame):
        text = "Press q to quit"
        cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


    def plot_bboxes(self, results, im0):
        class_ids = []
        self.annotator = Annotator(im0, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        probs = results[0].boxes.conf.cpu().tolist()  # Convert tensor to list
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        
        for box, prob, cls in zip(boxes, probs, clss):
            class_ids.append(cls)
            label = f'{names[int(cls)]} {prob:.2f}'
            self.annotator.box_label(box, label=label, color=colors(int(cls), True))
        return im0, class_ids
    
    def determine_winner(self, player_move, computer_move):
        if player_move == computer_move:
            return "Draw" , computer_move
        elif (player_move == "Rock" and computer_move == "Scissors") or \
             (player_move == "Scissors" and computer_move == "Paper") or \
             (player_move == "Paper" and computer_move == "Rock"):
            return "Player wins", computer_move
        else:
            return "Computer wins", computer_move

    def __call__(self):
        countdown = CountdownTimer()
        cap = cv2.VideoCapture(self.capture_index)
        # Check if the webcam is opened correctly
        assert cap.isOpened()
        # Set the video capture properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        game_text,computer_move_text = '',''
        # Read until video is completed
        while True:
           
            self.start_time = time.time()
            ret, frame = cap.read()
            assert ret

            # Generate computer's move randomly
            computer_move = random.choice(["Rock", "Paper", "Scissors"])
            

            results = self.predict(frame)
            frame, class_ids = self.plot_bboxes(results, frame)

            # self.display_fps(frame)
            self.display_info(frame)

            # Display the countdown on the frame if the timer has started
            if countdown.started:
                cv2.putText(frame, str(countdown.count), (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                countdown.start()

            # Reset the timer if it has finished
            if countdown.finished:
                count = 0
                if count >3:
                    break
                if len(class_ids) != 0:
                    player_move = results[0].names[int(class_ids[0])]
                    # print(player_move)
                    game_text, _ = self.determine_winner(player_move, computer_move)
                    # print(game_text)
                    
                    computer_move_text = f"Computer played {_}"
                    
                    
                else: 
                    game_text = "No move detected"
                    computer_move_text = ""
                    # print(game_text)
                countdown.start()    
                count +=1 

            cv2.putText(frame, game_text, (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(frame, computer_move_text, (400, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # Display the result of the game on the frame
            cv2.imshow('YOLOv8 Detection', frame)

            # Press 'q' to quit
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            elif key & 0xFF == ord('s'):
                countdown.start()
            
        # When everything done, release the video capture object
        cap.release()
        cv2.destroyAllWindows()
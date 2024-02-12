# Rock Paper Scissors with YOLO

Welcome to the Rock Paper Scissors with YOLO project! This project allows you to play the classic game of Rock Paper Scissors against your computer using your laptop's camera.

## Overview

This project utilizes the YOLO (You Only Look Once) object detection model to detect the hand gestures of Rock, Paper, and Scissors in real-time. The YOLO model has been fine-tuned on a custom dataset containing images of various hand gestures representing Rock, Paper, and Scissors.

## Features

- Real-time hand gesture detection using your laptop's camera.
- Computer opponent to play against.
- Simple and intuitive interface.

## Installation

To get started with the Rock Paper Scissors game, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/rock-paper-scissors-yolo.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script:

    ```bash
    python main.py
    ```

2. Follow the on-screen instructions to play the game. Make sure your laptop's camera is enabled and properly positioned.

3. Make your hand gestures to choose between Rock, Paper, or Scissors. The computer will also make its selection.

4. The winner of each round will be displayed on the screen, and the game will continue until you choose to quit.

## Customization

Feel free to customize the game according to your preferences. You can modify the YOLO model architecture, fine-tune it on a different dataset, or enhance the game interface.

## Credits

This project was inspired by the idea of combining computer vision techniques with classic games. Special thanks to the developers of YOLO for providing an excellent object detection framework.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

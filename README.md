
# Production 

https://probable-dog-airplane.vercel.app/

# CIFAR-10 Image Classification Web Interface

This project provides a **web interface** that allows users to upload images and get predictions from a **pre-trained CIFAR-10 classification model**. The model classifies images from the CIFAR-10 dataset, which consists of 10 different classes of images. The web app is built using **React**, and the predictions are powered by an **ONNX model**.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Web Interface](#web-interface)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project includes a **web interface** where users can upload images and get predictions from a **pre-trained CIFAR-10 model**. The web app is designed for easy interaction with the model, and TensorBoard is used for logging the training metrics of the model (for developers). The interface is built using **React** and communicates with a backend model (in ONNX format).

The app takes an image, preprocesses it, and uses the pre-trained model to classify the image. The results are shown to the user in a user-friendly manner.

---

## Installation

To set up the frontend locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/romainmltr/probable-dog-airplane.git
    cd probable-dog-airplane
    ```

2. Install the required dependencies:
    ```bash
    npm install
    ```

3. Run the app locally:
    ```bash
    npm start
    ```

4. Open your browser and go to:
    ```
    http://localhost:3000
    ```

---

## Usage

1. Open the web app in your browser.
2. Upload an image of a CIFAR-10 class '** "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck" ' **
3. The model will predict the class of the uploaded image and display the result.
4. If the model is not trained, you can replace the backend model or use a pre-trained ONNX model in the application.
5. 

---

## Web Interface

The **web interface** allows users to upload images and display the model's predictions. Here's how it works:

1. A user uploads an image via the file input.
2. The image is processed, and the pre-trained model makes predictions.
3. The predicted class is displayed on the webpage.
4. Use the assets in the folder assets to test online

---

## Contributing

Contributions are welcome! If you would like to contribute to this project, feel free to fork the repository, make improvements, and submit a pull request. For reporting issues or suggesting new features, open an issue in the GitHub repository.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

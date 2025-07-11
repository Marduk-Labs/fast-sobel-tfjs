# Sobel Edge Detection with React, TensorFlow.js, and Vite

This example demonstrates how to use the Sobel-TFJS library with a modern React application built using Vite and styled with Tailwind CSS.

## Features

- 📷 Upload images or use the provided sample image
- 🎚️ Configure kernel size, output format, and other filter options
- ⚡ Fast processing with TensorFlow.js
- 💾 Download the processed images
- 🎨 Modern UI built with Tailwind CSS

## Getting Started

First, make sure you have [Node.js](https://nodejs.org/) installed on your system.

Then, install the dependencies:

```bash
npm install
```

Next, start the development server:

```bash
npm run dev
```

The application should open in your browser at `http://localhost:5173`.

## Building for Production

To create a production build:

```bash
npm run build
```

The output will be in the `dist` directory.

## How It Works

This example demonstrates using the Sobel-TFJS library in a React application. The main components are:

- `App.tsx`: The main application component that initializes TensorFlow.js
- `ImageProcessor.tsx`: Handles image upload, processing, and display

The image processing workflow is:

1. User uploads an image or loads the sample image
2. The image is displayed on a canvas
3. User configures filter options
4. When the "Apply Sobel Filter" button is clicked, the Sobel filter is applied
5. The result is displayed on another canvas
6. User can download the original or processed image

## Configurable Options

- **Kernel Size**: Controls the size of the Sobel operator (3×3, 5×5, 7×7)
- **Output Format**: Choose between different gradient representations
- **Grayscale Conversion**: Option to convert the image to grayscale before processing
- **Scale Factor**: Adjust the intensity of the edge detection result

## Notes

- This example uses a local version of the Sobel-TFJS library (via `"file:../../"` in package.json)
- In a real-world application, you would install the library from npm 
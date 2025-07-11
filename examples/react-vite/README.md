# Fast Sobel TFJS - React Example

This example demonstrates how to use the **Fast Sobel TFJS** library with a modern React application built using Vite and styled with Tailwind CSS.

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

This example demonstrates using the **Fast Sobel TFJS** library in a React application. The main components are:

- `App.tsx`: The main application component that initializes TensorFlow.js
- `ImageProcessor.tsx`: Handles image upload, processing, and display
- `VideoProcessor.tsx`: Real-time webcam processing with live edge detection
- `BenchmarkPage.tsx`: Performance comparison between GPU and CPU implementations

The image processing workflow is:

1. User uploads an image or loads the sample image
2. The image is displayed on a canvas
3. User configures filter options (kernel size, output format, etc.)
4. When the "Apply Sobel Filter" button is clicked, the GPU-accelerated Sobel filter is applied
5. The result is displayed on another canvas with optional contrast enhancement
6. User can download the original or processed image

## Configurable Options

- **Kernel Size**: Controls the size of the Sobel operator (3×3, 5×5, 7×7)
- **Output Format**: Choose between magnitude, gradient, or normalized output
- **Grayscale Conversion**: Option to convert the image to grayscale before processing
- **Contrast Enhancement**: Boost edge visibility in the output
- **Scale Factor**: Adjust the intensity of the edge detection result

## Performance Features

- **GPU Acceleration**: 5-10x faster than CPU-only implementations
- **Real-time Video**: Live webcam processing at 30+ FPS
- **Benchmark Suite**: Compare performance across different image sizes
- **Memory Management**: Automatic tensor cleanup to prevent memory leaks

## Notes

- This example uses a local version of the Fast Sobel TFJS library (via `"file:../../"` in package.json)
- In a real-world application, you would install the library from npm: `npm install @marduk-labs/fast-sobel-tfjs`

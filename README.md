# 🚀 Fast Sobel TFJS

[![npm version](https://badge.fury.io/js/fast-sobel-tfjs.svg)](https://badge.fury.io/js/fast-sobel-tfjs)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=flat&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GPU-accelerated Sobel edge detection for images & video, powered by TensorFlow.js**

Blazing fast edge detection that runs on GPU via WebGL, with fallback to CPU. Perfect for real-time image processing, computer vision applications, and creative coding projects.

## 🎮 **Live Demo & Benchmark**

**📱 [Try the Interactive Demo](https://catorch.github.io/fast-sobel-tfjs/)**

Our live demo showcases all the capabilities of Fast Sobel TFJS:

- **🖼️ Image Processing**: Upload your own images and see real-time edge detection
- **📹 Video Processing**: Live webcam processing with adjustable parameters
- **⚡ Performance Benchmark**: Compare GPU vs CPU performance across different image sizes
- **🎛️ Interactive Controls**: Adjust kernel size, output format, and enhancement settings

**Benchmark Results**: See 5-10x performance improvements with GPU acceleration compared to CPU-only implementations.

## ⚡ Performance

- **5-10x faster** than CPU-only implementations
- **Real-time processing** for HD video (1080p @ 60fps)
- **WebGL acceleration** with automatic CPU fallback
- **Memory efficient** tensor operations with automatic cleanup
- **Zero copy** operations where possible

## 🎯 Features

- 🏃‍♂️ **GPU-accelerated** via TensorFlow.js WebGL backend
- 🖼️ **Multiple input formats**: ImageData, HTMLImageElement, HTMLVideoElement, Tensors
- 📱 **Cross-platform**: Works in browsers, Node.js, React Native
- 🎛️ **Configurable**: Multiple kernel sizes, output formats, and normalization options
- 🔧 **TypeScript**: Full type safety with comprehensive API
- 📦 **Lightweight**: ~30KB minified, peer dependency on TensorFlow.js

## 📦 Installation

```bash
npm install fast-sobel-tfjs @tensorflow/tfjs
```

**Peer Dependencies:**

- `@tensorflow/tfjs`: >=4.0.0

The library uses TensorFlow.js as a peer dependency to avoid bundle duplication and allow you to choose your preferred TF.js variant (CPU, WebGL, Node, etc.).

## 🚀 Quick Start

### Basic Edge Detection

```javascript
import { detectEdges } from "fast-sobel-tfjs";

// From image element
const img = document.getElementById("myImage");
const edges = await detectEdges(img);

// From canvas ImageData
const canvas = document.getElementById("myCanvas");
const ctx = canvas.getContext("2d");
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
const edgeData = await detectEdges(imageData);
```

### Advanced Usage with Custom Options

```javascript
import { SobelFilter } from "fast-sobel-tfjs";

const filter = new SobelFilter({
  kernelSize: 5, // 3, 5, or 7
  output: "gradient", // 'magnitude', 'gradient', 'normalized'
  normalizationRange: [0, 255],
  grayscale: true,
  threshold: 0.1,
});

// Process image
const result = await filter.processImage(imageElement);

// Or work with tensors directly
const tensor = tf.browser.fromPixels(imageElement);
const edges = filter.applyToTensor(tensor);
```

### Real-time Video Processing

```javascript
import { SobelFilter } from "fast-sobel-tfjs";

const video = document.getElementById("myVideo");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");

const filter = new SobelFilter({
  kernelSize: 3,
  output: "normalized",
  grayscale: true,
});

async function processFrame() {
  if (video.readyState === video.HAVE_ENOUGH_DATA) {
    const edges = await filter.processHTMLVideoElement(video);
    ctx.putImageData(edges, 0, 0);
  }
  requestAnimationFrame(processFrame);
}

processFrame();
```

## 📚 API Reference

### `detectEdges(input, useGrayscale?)`

Quick edge detection with optimal settings.

**Parameters:**

- `input`: `ImageData | HTMLImageElement | HTMLVideoElement | tf.Tensor3D`
- `useGrayscale`: `boolean` (default: `true`)

**Returns:** `Promise<ImageData | tf.Tensor3D>`

### `SobelFilter`

Main class for edge detection with full customization.

#### Constructor Options

```typescript
interface SobelOptions {
  kernelSize?: 3 | 5 | 7; // Default: 3
  output?: "magnitude" | "gradient" | "normalized"; // Default: 'magnitude'
  normalizationRange?: [number, number]; // Default: [0, 1]
  grayscale?: boolean; // Default: true
  threshold?: number; // Default: 0
}
```

#### Methods

##### `processImage(image)`

Process HTML image element.

- **Input:** `HTMLImageElement`
- **Returns:** `Promise<ImageData>`

##### `processImageData(imageData)`

Process canvas ImageData.

- **Input:** `ImageData`
- **Returns:** `Promise<ImageData>`

##### `processHTMLVideoElement(video)`

Process video element frame.

- **Input:** `HTMLVideoElement`
- **Returns:** `Promise<ImageData>`

##### `applyToTensor(tensor)`

Process tensor directly (advanced usage).

- **Input:** `tf.Tensor3D`
- **Returns:** `tf.Tensor3D`

### Output Formats

- **`'magnitude'`**: Grayscale edge strength
- **`'gradient'`**: RGB gradient components (Gx, Gy, magnitude)
- **`'normalized'`**: Normalized to specified range

### Utility Functions

```javascript
import {
  getAvailableKernelSizes,
  getAvailableOutputFormats,
  isValidKernelSize,
  isValidOutputFormat,
} from "fast-sobel-tfjs";
```

## 🎨 Examples

### Creative Effects

```javascript
// Artistic edge overlay
const filter = new SobelFilter({
  kernelSize: 7,
  output: "gradient",
  threshold: 0.2,
});

const edges = await filter.processImage(img);
// Blend with original image for artistic effect
```

### Medical Image Processing

```javascript
// High precision for medical imaging
const filter = new SobelFilter({
  kernelSize: 5,
  output: "magnitude",
  normalizationRange: [0, 4095], // 12-bit medical images
  grayscale: true,
});
```

### Real-time Webcam

```javascript
navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
  const video = document.createElement("video");
  video.srcObject = stream;
  video.play();

  const filter = new SobelFilter({ kernelSize: 3 });

  function processWebcam() {
    filter.processHTMLVideoElement(video).then((edges) => {
      // Display processed frame
      ctx.putImageData(edges, 0, 0);
      requestAnimationFrame(processWebcam);
    });
  }

  video.addEventListener("loadedmetadata", processWebcam);
});
```

## 🔧 Configuration

### Kernel Sizes

- **3x3**: Fastest, good for real-time applications
- **5x5**: Balanced performance and quality
- **7x7**: Highest quality, more computational cost

### Choosing Output Format

- **`'magnitude'`**: Best for edge detection and thresholding
- **`'gradient'`**: Best for directional analysis
- **`'normalized'`**: Best for display and further processing

### Performance Tips

1. **Use grayscale** when color information isn't needed
2. **Smaller kernel sizes** for real-time processing
3. **Batch processing** for multiple images
4. **Proper tensor disposal** to prevent memory leaks

```javascript
// Good: Automatic cleanup
const edges = await detectEdges(image);

// Advanced: Manual tensor management
tf.tidy(() => {
  const tensor = tf.browser.fromPixels(image);
  const edges = filter.applyToTensor(tensor);
  // Tensors automatically disposed at end of tidy
});
```

## 🌐 Browser Compatibility

- **Chrome**: 57+ (recommended)
- **Firefox**: 52+
- **Safari**: 11+
- **Edge**: 79+
- **Mobile**: iOS Safari 11+, Chrome Mobile 57+

**Requirements:**

- WebGL support for GPU acceleration
- ES2017+ or transpilation for older browsers

## 📊 Benchmarks

| Image Size | GPU Time | CPU Time | Speedup |
| ---------- | -------- | -------- | ------- |
| 640x480    | 2.1ms    | 12.8ms   | 6.1x    |
| 1280x720   | 4.3ms    | 31.2ms   | 7.3x    |
| 1920x1080  | 8.1ms    | 67.4ms   | 8.3x    |

_Benchmarks run on Chrome 120, RTX 3080, i7-12700K_

## 🛠️ Development

### Building

```bash
npm run build
```

### Testing

```bash
npm test
```

### Running Examples

```bash
# React example
npm run start:react
```

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.

### Development Setup

```bash
git clone https://github.com/catorch/fast-sobel-tfjs.git
cd fast-sobel-tfjs
npm install
npm run build
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- TensorFlow.js team for the amazing ML platform
- Computer vision researchers for Sobel operator algorithms
- Open source community for continuous improvements

---

<p align="center">
  <strong>Fast Sobel TFJS</strong> - GPU-accelerated edge detection for the modern web
</p>

<p align="center">
  Made with ❤️ by <a href="https://github.com/catorch">catorch</a>
</p>

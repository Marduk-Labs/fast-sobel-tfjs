import { ChangeEvent, useEffect, useRef, useState } from "react";
import { KernelSize, OutputFormat, SobelFilter } from "fast-sobel-tfjs";

// A simple embedded grayscale data URI of a placeholder image
const SAMPLE_IMAGE_URL = "/fast-sobel-tfjs/pugs.jpg";

// Helper function to enhance contrast of the result image
const enhanceContrast = (
  imageData: ImageData,
  factor: number = 2
): ImageData => {
  const data = imageData.data;
  const newData = new Uint8ClampedArray(data.length);

  // Find min and max values for auto-level
  let min = 255;
  let max = 0;

  // Only look at opacity or value data (every 4th value is alpha)
  for (let i = 0; i < data.length; i += 4) {
    const value = (data[i] + data[i + 1] + data[i + 2]) / 3;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }

  // Avoid division by zero
  const range = max - min || 1;

  // Apply contrast enhancement
  for (let i = 0; i < data.length; i += 4) {
    // Normalize to [0, 1], apply contrast, then scale back
    const r = (data[i] - min) / range;
    const g = (data[i + 1] - min) / range;
    const b = (data[i + 2] - min) / range;

    // Apply contrast curve (power function)
    newData[i] = Math.pow(r, 1 / factor) * 255;
    newData[i + 1] = Math.pow(g, 1 / factor) * 255;
    newData[i + 2] = Math.pow(b, 1 / factor) * 255;
    newData[i + 3] = data[i + 3]; // Keep original alpha
  }

  return new ImageData(newData, imageData.width, imageData.height);
};

const ImageProcessor = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [originalImage, setOriginalImage] = useState<HTMLImageElement | null>(
    null
  );
  const [originalImageUrl, setOriginalImageUrl] = useState<string>("");
  const [resultImageUrl, setResultImageUrl] = useState<string>("");
  const [fileName, setFileName] = useState<string>("");
  const [kernelSize, setKernelSize] = useState<KernelSize>(3);
  const [outputFormat, setOutputFormat] = useState<OutputFormat>("magnitude");
  const [useGrayscale, setUseGrayscale] = useState<boolean>(true);
  const [scale, setScale] = useState<number>(1.0);
  const [enhanceContrastEnabled, setEnhanceContrastEnabled] =
    useState<boolean>(true);

  const originalCanvasRef = useRef<HTMLCanvasElement>(null);
  const resultCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  // Initialize the result canvas
  useEffect(() => {
    const resultCanvas = resultCanvasRef.current;
    if (resultCanvas) {
      const ctx = resultCanvas.getContext("2d", { willReadFrequently: true });
      if (ctx) {
        // Set initial dimensions
        resultCanvas.width = 400;
        resultCanvas.height = 300;

        // Clear the canvas with a black background
        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, resultCanvas.width, resultCanvas.height);
      }
    }
  }, []);

  // Automatically load the sample image
  useEffect(() => {
    loadSampleImage();
  }, []);

  // Handle file selection
  const handleFileSelect = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setFileName(file.name);

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      loadImage(dataUrl);
    };
    reader.readAsDataURL(file);
  };

  // Load an image from URL or data URL
  const loadImage = (url: string) => {
    const img = new Image();
    img.crossOrigin = "anonymous";

    img.onload = () => {
      setOriginalImage(img);
      setOriginalImageUrl(url);

      // Clear previous result
      setResultImageUrl("");

      // Draw original image on canvas
      if (originalCanvasRef.current) {
        const canvas = originalCanvasRef.current;
        canvas.width = img.width;
        canvas.height = img.height;

        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (ctx) {
          ctx.drawImage(img, 0, 0);
        }
      }
    };

    img.src = url;
  };

  // Add this function to explicitly create and load a sample image
  const createSampleImage = () => {
    const img = new Image();
    img.crossOrigin = "anonymous";

    // Use a data URI for a grayscale gradient pattern if the sample image isn't loading
    img.onerror = () => {
      console.warn("Sample image failed to load, using fallback pattern");
      // Create a simple gradient canvas
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = 400;
      tempCanvas.height = 300;
      const ctx = tempCanvas.getContext("2d");

      if (ctx) {
        // Create a gradient
        const gradient = ctx.createLinearGradient(
          0,
          0,
          tempCanvas.width,
          tempCanvas.height
        );
        gradient.addColorStop(0, "#333");
        gradient.addColorStop(1, "#999");

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

        // Add some shapes for edge detection
        ctx.fillStyle = "#000";
        ctx.fillRect(50, 50, 100, 100);
        ctx.fillRect(250, 150, 100, 100);

        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(200, 150, 80, 0, Math.PI * 2);
        ctx.fill();

        // Convert canvas to image
        const dataUrl = tempCanvas.toDataURL();
        img.src = dataUrl;
      }
    };

    img.onload = () => {
      setOriginalImage(img);
      setOriginalImageUrl(img.src);

      // Draw original image on canvas
      if (originalCanvasRef.current) {
        const canvas = originalCanvasRef.current;
        canvas.width = img.width;
        canvas.height = img.height;

        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (ctx) {
          ctx.drawImage(img, 0, 0);
        }
      }
    };

    img.src = SAMPLE_IMAGE_URL;
    setFileName("sample.jpg");
  };

  const loadSampleImage = () => {
    const imgPath = SAMPLE_IMAGE_URL;
    console.log(`[IMAGE] Loading sample image from: ${imgPath}`);

    // Load the image directly
    const img = new Image();
    img.crossOrigin = "anonymous";

    // Handle loading errors by falling back to createSampleImage
    img.onerror = () => {
      console.warn(
        `[IMAGE] Failed to load sample from ${imgPath}, falling back to generated image`
      );
      createSampleImage();
    };

    img.onload = () => {
      console.log(
        `[IMAGE] Sample image loaded successfully: ${img.width}x${img.height}`
      );
      setOriginalImage(img);
      setOriginalImageUrl(img.src);
      setFileName("pugs.jpg");

      // Draw original image on canvas
      if (originalCanvasRef.current) {
        const canvas = originalCanvasRef.current;
        canvas.width = img.width;
        canvas.height = img.height;

        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        if (ctx) {
          ctx.drawImage(img, 0, 0);
        }
      }
    };

    img.src = imgPath;
  };

  // Process the image with Sobel filter
  const processImage = async () => {
    if (!originalImage) return;

    setIsProcessing(true);

    try {
      // Initialize the result canvas if it doesn't exist
      const resultCanvas = resultCanvasRef.current;
      if (!resultCanvas) {
        console.error("Result canvas is not available");
        alert(
          "Error: Result canvas not available. Please try reloading the page."
        );
        return;
      }

      // Get the image data from the original canvas
      const canvas = originalCanvasRef.current;
      if (!canvas) throw new Error("Canvas not available");

      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) throw new Error("Canvas context not available");

      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

      console.log(
        "Original Image data dimensions:",
        imageData.width,
        "x",
        imageData.height
      );

      // Debug - Check if the original data has actual values
      const origDataArray = new Uint8Array(imageData.data.buffer);
      const origSum = origDataArray.reduce((acc, val) => acc + val, 0);
      console.log(
        `Original image data sum: ${origSum}, length: ${origDataArray.length}`
      );

      if (origSum === 0) {
        console.error(
          "ERROR: Original image data is all zeros. Canvas might be empty."
        );
        alert(
          "The canvas appears to be empty. Please try reloading the image."
        );
        setIsProcessing(false);
        return;
      }

      // Create a SobelFilter instance with the selected options
      console.log("Initializing SobelFilter with settings:", {
        kernelSize,
        output: outputFormat,
        grayscale: useGrayscale,
        normalizationRange: [0, 255],
      });

      const sobelFilter = new SobelFilter({
        kernelSize,
        output: outputFormat,
        grayscale: useGrayscale,
        normalizationRange: [0, 255] as [number, number],
      });

      // Process image using the Sobel-TFJS library
      console.log("Using Sobel-TFJS library");
      console.time("sobelLibrary");

      try {
        let resultImageData = await sobelFilter.processImageData(imageData);
        console.timeEnd("sobelLibrary");
        console.log("Sobel-TFJS library processing successful");

        // Log stats *before* post-processing
        let dataArray = new Uint8Array(resultImageData.data.buffer);
        let sum = dataArray.reduce((acc, val) => acc + val, 0);
        console.log(
          `Processed image data sum (BEFORE post-processing): ${sum}`
        );

        // Compute some statistics
        let min = 255,
          max = 0,
          nonZeroPixels = 0;
        for (let i = 0; i < dataArray.length; i += 4) {
          const avg = (dataArray[i] + dataArray[i + 1] + dataArray[i + 2]) / 3;
          min = Math.min(min, avg);
          max = Math.max(max, avg);
          if (avg > 0) nonZeroPixels++;
        }
        console.log(
          `Image stats (BEFORE post-proc) - Min: ${min}, Max: ${max}, Non-zero pixels: ${nonZeroPixels} (${(
            (nonZeroPixels / (dataArray.length / 4)) *
            100
          ).toFixed(2)}%)`
        );

        if (sum === 0) {
          console.warn(
            "WARNING: All pixel values are 0. The result may appear black."
          );
          alert(
            "The processed image appears to be black. Please try a different image or settings."
          );
          setIsProcessing(false);
          return;
        }

        // Enhance contrast if enabled
        if (enhanceContrastEnabled) {
          console.log("Enhancing contrast...");
          resultImageData = enhanceContrast(resultImageData, 1.5);
          // Log sum *after* contrast
          dataArray = new Uint8Array(resultImageData.data.buffer);
          sum = dataArray.reduce((acc, val) => acc + val, 0);
          console.log(`Processed image data sum (AFTER contrast): ${sum}`);
        }

        // Apply scaling if needed
        if (scale !== 1.0 && scale !== 0) {
          console.log(`Applying scaling with factor: ${scale}`);
          // Create a temporary canvas to apply scaling
          const tempCanvas = document.createElement("canvas");
          tempCanvas.width = canvas.width;
          tempCanvas.height = canvas.height;

          const tempCtx = tempCanvas.getContext("2d", {
            willReadFrequently: true,
          });
          if (tempCtx) {
            // Draw the initial result
            tempCtx.putImageData(resultImageData, 0, 0);

            // Create a new canvas for the scaled result
            const scaledCanvas = document.createElement("canvas");
            scaledCanvas.width = canvas.width;
            scaledCanvas.height = canvas.height;

            const scaledCtx = scaledCanvas.getContext("2d", {
              willReadFrequently: true,
            });
            if (scaledCtx) {
              // Apply scaling with a composite operation
              scaledCtx.fillStyle = "black"; // Change to black background
              scaledCtx.fillRect(0, 0, canvas.width, canvas.height);
              scaledCtx.globalAlpha = scale;
              scaledCtx.drawImage(tempCanvas, 0, 0);

              // Get the result data
              resultImageData = scaledCtx.getImageData(
                0,
                0,
                canvas.width,
                canvas.height
              );
              // Log sum *after* scaling
              dataArray = new Uint8Array(resultImageData.data.buffer);
              sum = dataArray.reduce((acc, val) => acc + val, 0);
              console.log(`Processed image data sum (AFTER scaling): ${sum}`);
            }
          }
        }

        // Log final data before drawing
        const finalDataArray = new Uint8Array(resultImageData.data.buffer);
        const finalSum = finalDataArray.reduce((acc, val) => acc + val, 0);
        console.log(`Final image data sum (BEFORE drawing): ${finalSum}`);
        console.log(
          `Final ImageData dimensions: ${resultImageData.width}x${resultImageData.height}`
        );

        // Validate ImageData with a temporary canvas first
        console.log("Validating ImageData with temporary canvas...");
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = resultImageData.width;
        tempCanvas.height = resultImageData.height;
        const tempCtx = tempCanvas.getContext("2d");
        if (tempCtx) {
          tempCtx.putImageData(resultImageData, 0, 0);
          const tempUrl = tempCanvas.toDataURL();
          console.log("Temporary canvas URL generated");

          // Check if the temporary canvas shows the image
          const testImg = new Image();
          testImg.onload = () =>
            console.log("Validation image loaded successfully");
          testImg.onerror = (err) =>
            console.error("Validation image failed to load:", err);
          testImg.src = tempUrl;
        }

        // Update the main result canvas
        console.log("Preparing main result canvas for display...");
        const resultCanvas = resultCanvasRef.current;
        if (!resultCanvas) throw new Error("Result canvas ref not available");

        // Ensure canvas size matches image data
        if (
          resultCanvas.width !== resultImageData.width ||
          resultCanvas.height !== resultImageData.height
        ) {
          console.log(
            `Resizing result canvas from ${resultCanvas.width}x${resultCanvas.height} to ${resultImageData.width}x${resultImageData.height}`
          );
          resultCanvas.width = resultImageData.width;
          resultCanvas.height = resultImageData.height;
        }

        // Get context with alpha enabled
        const resultCtx = resultCanvas.getContext("2d", {
          alpha: true,
          willReadFrequently: true, // Need true to read back for debugging/URL
        });
        if (!resultCtx) throw new Error("Result canvas context not available");

        console.log("Attempting to premultiply alpha...");
        const premultipliedData = new Uint8ClampedArray(
          resultImageData.data.length
        );
        let nonZeroAlphaCount = 0;
        for (let i = 0; i < resultImageData.data.length; i += 4) {
          const alpha = resultImageData.data[i + 3] / 255;
          if (alpha > 0) nonZeroAlphaCount++;
          premultipliedData[i] = Math.round(resultImageData.data[i] * alpha);
          premultipliedData[i + 1] = Math.round(
            resultImageData.data[i + 1] * alpha
          );
          premultipliedData[i + 2] = Math.round(
            resultImageData.data[i + 2] * alpha
          );
          premultipliedData[i + 3] = resultImageData.data[i + 3];
        }
        console.log(
          `Premultiplied alpha for ${nonZeroAlphaCount} pixels with non-zero alpha.`
        );
        const premultipliedImageData = new ImageData(
          premultipliedData,
          resultImageData.width,
          resultImageData.height
        );
        // --- End Premultiply Alpha ---

        // --- Additional Debugging ---
        console.log(
          "First 40 pixels of PREMULTIPLIED ImageData:",
          Array.from(premultipliedImageData.data.slice(0, 40))
        );

        const testCanvas = document.createElement("canvas");
        testCanvas.width = 10;
        testCanvas.height = 10;
        const testCtx = testCanvas.getContext("2d");
        if (testCtx) {
          try {
            testCtx.putImageData(
              new ImageData(
                new Uint8ClampedArray(
                  premultipliedImageData.data.slice(0, 400)
                ),
                10,
                10
              ),
              0,
              0
            );
            console.log(
              "Test canvas (10x10) URL:",
              testCanvas.toDataURL().substring(0, 100) + "..."
            );
          } catch (e) {
            console.error("Error creating test canvas URL:", e);
          }
        }
        // --- End Additional Debugging ---

        // Clear canvas and draw directly using the PREMULTIPLIED data
        console.log(
          "Clearing and drawing PREMULTIPLIED data to main canvas..."
        );
        resultCtx.clearRect(0, 0, resultCanvas.width, resultCanvas.height);
        resultCtx.putImageData(premultipliedImageData, 0, 0); // Use premultiplied data

        // Verify the drawn data
        try {
          const checkData = resultCtx.getImageData(
            0,
            0,
            resultCanvas.width,
            resultCanvas.height
          );
          const checkSum = Array.from(checkData.data).reduce(
            (sum, val) => sum + val,
            0
          );
          console.log(
            `Main canvas sum (AFTER drawing PREMULTIPLIED): ${checkSum}`
          );
        } catch (readbackError: any) {
          console.warn(
            `Error reading back main canvas: ${readbackError.message}`
          );
        }

        // Generate and set the data URL from the main canvas
        console.log("Generating data URL from main canvas...");
        const dataURL = resultCanvas.toDataURL();
        console.log(
          "Setting result image URL, canvas dimensions:",
          resultCanvas.width,
          "x",
          resultCanvas.height
        );
        setResultImageUrl(dataURL);

        // Ensure the canvas is visible
        resultCanvas.style.display = "block";
        console.log("Result canvas display style:", resultCanvas.style.display);
      } catch (error: any) {
        console.error("Error during Sobel processing:", error);
        alert(`Error during Sobel processing: ${error.message}`);
      }
    } catch (error) {
      console.error("Error processing image:", error);
      alert("An error occurred during processing: " + error);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle drag and drop
  useEffect(() => {
    const dropZone = dropZoneRef.current;
    if (!dropZone) return;

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault();
      dropZone.classList.add("border-accent");
    };

    const handleDragLeave = () => {
      dropZone.classList.remove("border-accent");
    };

    const handleDrop = (e: DragEvent) => {
      e.preventDefault();
      dropZone.classList.remove("border-accent");

      if (e.dataTransfer?.files.length) {
        const file = e.dataTransfer.files[0];
        if (file.type.startsWith("image/")) {
          setFileName(file.name);

          const reader = new FileReader();
          reader.onload = (e) => {
            const dataUrl = e.target?.result as string;
            loadImage(dataUrl);
          };
          reader.readAsDataURL(file);
        }
      }
    };

    dropZone.addEventListener("dragover", handleDragOver);
    dropZone.addEventListener("dragleave", handleDragLeave);
    dropZone.addEventListener("drop", handleDrop);

    return () => {
      dropZone.removeEventListener("dragover", handleDragOver);
      dropZone.removeEventListener("dragleave", handleDragLeave);
      dropZone.removeEventListener("drop", handleDrop);
    };
  }, []);

  // Download processed image
  const downloadImage = (
    canvas: HTMLCanvasElement | null,
    filename: string
  ) => {
    if (!canvas) return;

    const link = document.createElement("a");
    link.download = filename;
    link.href = canvas.toDataURL("image/png");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Monitor result image URL changes
  useEffect(() => {
    if (resultImageUrl) {
      console.log(
        "Result image URL updated:",
        resultImageUrl.substring(0, 50) + "..."
      );

      // Ensure canvas is visible
      const resultCanvas = resultCanvasRef.current;
      if (resultCanvas) {
        resultCanvas.style.display = "block";

        // Log canvas dimensions to help debug
        console.log(
          "Result canvas dimensions:",
          resultCanvas.width,
          "x",
          resultCanvas.height
        );
      }
    }
  }, [resultImageUrl]);

  return (
    <div className="max-w-6xl mx-auto">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Left column: Upload and options */}
        <div className="space-y-6">
          {/* File upload */}
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <h2 className="text-xl font-semibold mb-4">Upload Image</h2>

            <div
              ref={dropZoneRef}
              className="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center cursor-pointer transition-colors duration-200 mb-4"
              onClick={() => fileInputRef.current?.click()}
            >
              <svg
                className="w-12 h-12 mx-auto mb-2 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <p className="text-gray-400">
                Drag & drop an image here, or click to browse
              </p>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileSelect}
              />
              {fileName && <p className="mt-2 text-accent-light">{fileName}</p>}
            </div>

            <button
              onClick={loadSampleImage}
              className="w-full py-2 px-4 bg-gray-800 text-gray-200 rounded hover:bg-gray-700 transition-colors"
            >
              Load Sample Image
            </button>
          </div>

          {/* Options */}
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <h2 className="text-xl font-semibold mb-4">Filter Options</h2>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">
                  Kernel Size
                </label>
                <select
                  value={kernelSize}
                  onChange={(e) =>
                    setKernelSize(Number(e.target.value) as KernelSize)
                  }
                  className="w-full bg-gray-800 text-white rounded p-2 border border-gray-700"
                >
                  <option value={3}>3×3 (Standard)</option>
                  <option value={5}>5×5 (More detail)</option>
                  <option value={7}>7×7 (High detail)</option>
                </select>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-1">
                  Output Format
                </label>
                <select
                  value={outputFormat}
                  onChange={(e) =>
                    setOutputFormat(e.target.value as OutputFormat)
                  }
                  className="w-full bg-gray-800 text-white rounded p-2 border border-gray-700"
                >
                  <option value="magnitude">Magnitude (Default)</option>
                  <option value="x">Horizontal Edges</option>
                  <option value="y">Vertical Edges</option>
                  <option value="direction">Direction</option>
                  <option value="normalized">Normalized</option>
                </select>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="grayscale"
                  checked={useGrayscale}
                  onChange={(e) => setUseGrayscale(e.target.checked)}
                  className="mr-2 h-4 w-4"
                />
                <label htmlFor="grayscale" className="text-gray-300">
                  Convert to grayscale
                </label>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="enhanceContrast"
                  checked={enhanceContrastEnabled}
                  onChange={(e) => setEnhanceContrastEnabled(e.target.checked)}
                  className="mr-2 h-4 w-4"
                />
                <label htmlFor="enhanceContrast" className="text-gray-300">
                  Enhance contrast
                </label>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-1">
                  Scale Factor: {scale.toFixed(1)}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="5"
                  step="0.1"
                  value={scale}
                  onChange={(e) => setScale(parseFloat(e.target.value))}
                  className="w-full"
                />
              </div>

              <button
                onClick={processImage}
                disabled={!originalImage || isProcessing}
                className={`w-full py-3 px-4 rounded font-medium transition-colors ${
                  !originalImage || isProcessing
                    ? "bg-gray-700 text-gray-300 cursor-not-allowed"
                    : "bg-accent hover:bg-accent-light text-white"
                }`}
              >
                {isProcessing ? (
                  <span className="flex items-center justify-center">
                    <svg
                      className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Processing...
                  </span>
                ) : (
                  "Apply Sobel Filter"
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right column: Image previews */}
        <div className="space-y-6">
          {/* Original image */}
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Original Image</h2>
              {originalImageUrl && (
                <button
                  onClick={() =>
                    downloadImage(originalCanvasRef.current, "original.png")
                  }
                  className="text-accent hover:text-accent-light"
                >
                  Download
                </button>
              )}
            </div>

            <div className="bg-black rounded-lg overflow-hidden flex items-center justify-center">
              {originalImageUrl ? (
                <canvas
                  ref={originalCanvasRef}
                  className="max-w-full max-h-[400px] object-contain"
                />
              ) : (
                <div className="p-8 text-gray-500">No image uploaded</div>
              )}
            </div>
          </div>

          {/* Result image */}
          <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Edge Detection Results</h2>
              <div>
                {resultImageUrl && (
                  <button
                    onClick={() =>
                      downloadImage(
                        resultCanvasRef.current,
                        "edge-detection-tfjs.png"
                      )
                    }
                    className="text-accent hover:text-accent-light"
                  >
                    Download Result
                  </button>
                )}
              </div>
            </div>

            {/* TensorFlow.js Result */}
            <div className="bg-black rounded-lg overflow-hidden relative min-h-[200px]">
              {/* Canvas container */}
              <div className="w-full h-full flex items-center justify-center">
                <canvas
                  ref={resultCanvasRef}
                  className="max-w-full max-h-[300px] object-contain"
                  style={{
                    display: resultImageUrl && !isProcessing ? "block" : "none",
                  }}
                />
                {resultImageUrl && !isProcessing && (
                  <img
                    src={resultImageUrl}
                    alt="TensorFlow.js edge detection"
                    className="max-w-full max-h-[300px] object-contain absolute top-0 left-0 w-full h-full opacity-0 hover:opacity-100"
                    style={{ transition: "opacity 0.3s" }}
                    title="Hover to see direct image result"
                  />
                )}
              </div>

              {isProcessing ? (
                <div className="absolute inset-0 flex items-center justify-center z-20 bg-black bg-opacity-50">
                  <div className="text-center">
                    <div className="inline-block h-10 w-10 animate-spin rounded-full border-4 border-accent border-r-transparent"></div>
                    <p className="mt-4 text-gray-400">Processing...</p>
                  </div>
                </div>
              ) : !resultImageUrl ? (
                <div className="absolute inset-0 flex items-center justify-center z-20">
                  <div className="p-8 text-gray-500 text-center">
                    {originalImageUrl
                      ? 'Click "Apply Sobel Filter" to process the image'
                      : "Upload an image to get started"}
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageProcessor;

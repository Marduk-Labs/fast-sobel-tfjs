import * as tf from '@tensorflow/tfjs';
import { useEffect, useRef, useState } from 'react';
import { KernelSize, OutputFormat, SobelFilter } from 'sobel-tfjs';

const VideoProcessor = () => {
    const [isProcessing, setIsProcessing] = useState(false);
    const [isStreaming, setIsStreaming] = useState(false);
    const [kernelSize, setKernelSize] = useState<KernelSize>(3);
    const [outputFormat, setOutputFormat] = useState<OutputFormat>('magnitude');
    const [useGrayscale, setUseGrayscale] = useState<boolean>(true);
    const [enhanceContrastEnabled, setEnhanceContrastEnabled] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const [isVisible, setIsVisible] = useState<boolean>(true);
    const [fps, setFps] = useState<number>(0);
    const [processingTime, setProcessingTime] = useState<number>(0);
    const [frameCount, setFrameCount] = useState<number>(0);
    const [lowQualityMode, setLowQualityMode] = useState<boolean>(false);

    const videoRef = useRef<HTMLVideoElement>(null);
    const resultCanvasRef = useRef<HTMLCanvasElement>(null);
    const requestIdRef = useRef<number | null>(null);
    const lastFrameTimeRef = useRef<number>(0);
    const frameTimesRef = useRef<number[]>([]);

    // Sobel filter instance
    const sobelFilterRef = useRef<SobelFilter | null>(null);

    // Handle document visibility change
    useEffect(() => {
        const handleVisibilityChange = () => {
            const isDocumentVisible = document.visibilityState === 'visible';
            console.log(`[VISIBILITY] Changed to: ${isDocumentVisible}`);
            setIsVisible(isDocumentVisible);

            if (isStreaming) {
                if (isDocumentVisible && requestIdRef.current === null) {
                    // Resume processing only if it was stopped due to visibility
                    console.log("[VISIBILITY] Resuming processing");
                    startProcessing();
                } else if (!isDocumentVisible && requestIdRef.current !== null) {
                    // Pause processing when tab is not visible
                    console.log("[VISIBILITY] Pausing processing");
                    stopProcessing();
                }
            }
        };

        handleVisibilityChange(); // Initial check
        document.addEventListener('visibilitychange', handleVisibilityChange);
        return () => {
            document.removeEventListener('visibilitychange', handleVisibilityChange);
        };
    }, [isStreaming]); // Depend only on isStreaming

    // Initialize the Sobel filter
    useEffect(() => {
        try {
            console.log("[SOBEL-INIT] Creating/Updating filter with:",
                { kernelSize, outputFormat, useGrayscale }
            );
            sobelFilterRef.current = new SobelFilter({
                kernelSize,
                output: outputFormat,
                grayscale: useGrayscale
            });
            console.log("[SOBEL-INIT] Filter updated successfully");
        } catch (err) {
            console.error("[SOBEL-INIT] Error initializing filter:", err);
            setError('Failed to initialize edge detection filter');
        }
    }, [kernelSize, outputFormat, useGrayscale]);

    // Start the webcam stream
    const startStream = async () => {
        try {
            console.log("[WEBCAM] Requesting camera access");
            setError(null);
            setFrameCount(0); // Reset frame count
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                }
            };

            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            console.log("[WEBCAM] Camera access granted");

            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                console.log("[WEBCAM] Set stream to video element");
                videoRef.current.onloadedmetadata = () => {
                    videoRef.current?.play().then(() => {
                        console.log("[WEBCAM] Video playback started");
                        setIsStreaming(true);
                        startProcessing(); // Start the processing loop
                    }).catch(playErr => {
                        console.error("[WEBCAM] Error starting video playback:", playErr);
                        setError("Failed to start video playback.");
                    });
                };
            }
        } catch (err) {
            console.error("[WEBCAM] Error accessing webcam:", err);
            setError("Could not access webcam. Check permissions.");
        }
    };

    // Stop the webcam stream
    const stopStream = () => {
        stopProcessing(); // Stop the processing loop first
        if (videoRef.current && videoRef.current.srcObject) {
            console.log("[WEBCAM] Stopping stream tracks");
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
            setIsStreaming(false);
        }
    };

    // Start processing frames
    const startProcessing = () => {
        if (requestIdRef.current === null) { // Check if already running
            setIsProcessing(true); // For UI indication
            console.log("[PROCESS] Starting animation loop");
            // Reset FPS calculation state
            lastFrameTimeRef.current = performance.now();
            frameTimesRef.current = [];
            // Schedule the first frame
            requestIdRef.current = requestAnimationFrame(processFrame);
        } else {
            console.log("[PROCESS] Animation loop already running");
        }
    };

    // Stop processing frames
    const stopProcessing = () => {
        if (requestIdRef.current !== null) {
            console.log("[PROCESS] Stopping animation loop");
            cancelAnimationFrame(requestIdRef.current);
            requestIdRef.current = null;
        }
        setIsProcessing(false); // Update UI state
    };

    // Helper function to enhance contrast of the result image
    const enhanceContrast = (imageData: ImageData, factor: number = 2): ImageData => {
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
            const r = ((data[i] - min) / range);
            const g = ((data[i + 1] - min) / range);
            const b = ((data[i + 2] - min) / range);

            // Apply contrast curve (power function)
            newData[i] = Math.pow(r, 1 / factor) * 255;
            newData[i + 1] = Math.pow(g, 1 / factor) * 255;
            newData[i + 2] = Math.pow(b, 1 / factor) * 255;
            newData[i + 3] = data[i + 3]; // Keep original alpha
        }

        return new ImageData(newData, imageData.width, imageData.height);
    };

    // Helper function to get a downscaled tensor from video for better performance
    const getVideoTensor = (video: HTMLVideoElement, lowQuality: boolean): tf.Tensor3D => {
        const width = video.videoWidth;
        const height = video.videoHeight;
        console.log(`[TENSOR] getVideoTensor called. Video: ${width}x${height}, lowQuality: ${lowQuality}`);

        return tf.tidy(() => {
            // Get tensor from video element
            const fullResTensor = tf.browser.fromPixels(video);
            console.log(`[TENSOR]   Full res tensor shape: [${fullResTensor.shape}]`);

            // If low quality, resize to smaller dimensions for better performance
            if (lowQuality && width > 320 && height > 240) {
                console.log(`[TENSOR]   Applying low quality resize...`);
                // Calculate aspect ratio
                const aspectRatio = width / height;
                let newWidth = 320;
                let newHeight = Math.round(newWidth / aspectRatio);

                // Ensure height is also capped
                if (newHeight > 240) {
                    newHeight = 240;
                    newWidth = Math.round(newHeight * aspectRatio);
                }

                console.log(`[TENSOR]   Downscaling from ${width}x${height} to ${newWidth}x${newHeight}`);

                // Resize the tensor to lower resolution
                const resizedTensor = tf.image.resizeBilinear(fullResTensor, [newHeight, newWidth]) as tf.Tensor3D;
                console.log(`[TENSOR]   Resized tensor shape: [${resizedTensor.shape}]`);
                return resizedTensor;
            } else {
                console.log(`[TENSOR]   Using full resolution tensor (lowQuality=${lowQuality}, width=${width}, height=${height})`);
                // Return full resolution tensor
                return fullResTensor;
            }
        });
    };

    // Process video frame
    const processFrame = () => {
        // --- Synchronous Checks --- 
        if (requestIdRef.current === null || !isVisible || !videoRef.current || !resultCanvasRef.current) {
            return;
        }

        const video = videoRef.current;
        const resultCanvas = resultCanvasRef.current;

        // Video readiness check
        if (video.readyState < video.HAVE_ENOUGH_DATA || video.videoWidth === 0 || video.videoHeight === 0) {
            requestIdRef.current = requestAnimationFrame(processFrame); // Try again next frame
            return;
        }

        // --- Synchronous Setup (Canvas Resizing Removed Here) --- 
        // Get result context FIRST - needed for fallback drawing
        const resultCtx = resultCanvas.getContext('2d', { willReadFrequently: true });
        if (!resultCtx) {
            console.error("[FRAME] Failed to get 2D context from result canvas. Stopping processing.");
            stopProcessing(); // Stop the loop if context fails
            setError("Canvas context error. Cannot display results.");
            return;
        }

        // Update FPS (can be done synchronously)
        const now = performance.now();
        const elapsed = now - lastFrameTimeRef.current;
        lastFrameTimeRef.current = now;
        frameTimesRef.current.push(elapsed);
        if (frameTimesRef.current.length > 10) frameTimesRef.current.shift();
        const avgFrameTime = frameTimesRef.current.reduce((a, b) => a + b, 0) / frameTimesRef.current.length;
        setFps(Math.round(1000 / avgFrameTime));

        // --- Asynchronous Processing --- 
        (async () => {
            let inputTensor: tf.Tensor3D | null = null;
            try {
                const startTime = performance.now();

                // Ensure we have a filter
                if (!sobelFilterRef.current) {
                    console.warn("[PROCESS] Sobel filter missing, creating new one");
                    sobelFilterRef.current = new SobelFilter({ kernelSize, output: outputFormat, grayscale: useGrayscale });
                }

                // Create input tensor (handles low quality mode)
                inputTensor = getVideoTensor(video, lowQualityMode);
                const [tensorHeight, tensorWidth] = inputTensor.shape;

                // --- Set Canvas Size to Match Tensor --- 
                if (resultCanvas.width !== tensorWidth || resultCanvas.height !== tensorHeight) {
                    console.log(`[FRAME] Setting result canvas dimensions to match tensor: ${tensorWidth}x${tensorHeight}`);
                    resultCanvas.width = tensorWidth;
                    resultCanvas.height = tensorHeight;
                }
                // -------------------------------------

                // Apply Sobel filter (now returns normalized tensor)
                const resultTensor = sobelFilterRef.current.applyToTensor(inputTensor);

                // Render the directly returned (and normalized) result tensor to canvas
                await tf.browser.toPixels(resultTensor, resultCanvas);

                // Enhance contrast if enabled (operates on the canvas now)
                if (enhanceContrastEnabled) {
                    // Get context *after* potential resize
                    const currentResultCtx = resultCanvas.getContext('2d', { willReadFrequently: true });
                    if (currentResultCtx) {
                        const imageData = currentResultCtx.getImageData(0, 0, resultCanvas.width, resultCanvas.height);
                        const enhancedData = enhanceContrast(imageData);
                        currentResultCtx.putImageData(enhancedData, 0, 0);
                    } else {
                        console.warn("[ENHANCE] Could not get context after resize for enhancement");
                    }
                }

                // Clean up result tensor immediately
                resultTensor.dispose();

                // Update metrics
                const endTime = performance.now();
                const totalTime = endTime - startTime;
                setProcessingTime(totalTime);
                setFrameCount(prev => prev + 1);
                setError(null); // Clear previous errors on success

            } catch (err: unknown) {
                console.error("[PROCESS-ASYNC] Error during processing:", err);
                // Use the context obtained earlier for fallback drawing
                resultCtx.drawImage(video, 0, 0, resultCanvas.width, resultCanvas.height); // Draw to current canvas size
                if (!error) {
                    const message = err instanceof Error ? err.message : String(err);
                    setError(`Processing error: ${message.substring(0, 100)}`);
                }
            } finally {
                // Always clean up input tensor if it was created
                if (inputTensor && !inputTensor.isDisposed) {
                    inputTensor.dispose();
                }
                // --- Schedule Next Frame --- 
                if (requestIdRef.current !== null) {
                    requestIdRef.current = requestAnimationFrame(processFrame);
                }
            }
        })(); // Immediately invoke the async part
    };

    // Clean up on unmount
    useEffect(() => {
        return () => {
            console.log("[UNMOUNT] Cleaning up VideoProcessor");
            stopStream(); // Ensure stream and loop are stopped
        };
    }, []);

    return (
        <div className="max-w-6xl mx-auto px-4">
            <div className="flex flex-col md:flex-row gap-4 md:gap-8">
                <div className="flex-1 space-y-4">
                    <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
                        <h2 className="text-xl font-semibold text-gray-200 mb-4">Camera Input</h2>
                        <div className="aspect-video bg-gray-700 rounded flex items-center justify-center relative overflow-hidden">
                            <video
                                ref={videoRef}
                                className="max-w-full max-h-full object-contain"
                                muted
                                playsInline
                            />
                            {!isStreaming && (
                                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70">
                                    <button
                                        onClick={startStream}
                                        className="px-4 py-2 bg-accent text-white rounded shadow hover:bg-accent-light transition-colors"
                                    >
                                        Start Camera
                                    </button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                <div className="flex-1 space-y-4">
                    <div className="bg-gray-800 p-4 rounded-lg shadow-lg">
                        <h2 className="text-xl font-semibold text-gray-200 mb-4">Edge Detection Result</h2>
                        <div className="aspect-video bg-gray-700 rounded flex items-center justify-center overflow-hidden relative">
                            <canvas
                                ref={resultCanvasRef}
                                className="max-w-full max-h-full object-contain"
                            />
                            {isStreaming && frameCount === 0 && !error && (
                                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70">
                                    <div className="flex flex-col items-center">
                                        <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-accent border-r-transparent mb-2"></div>
                                        <p className="text-white">Processing first frame...</p>
                                    </div>
                                </div>
                            )}
                            {error && (
                                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70">
                                    <p className="text-red-500 bg-gray-900 p-4 rounded-lg text-sm">Error: {error}</p>
                                </div>
                            )}
                        </div>
                        {isStreaming && frameCount > 0 && (
                            <div className="mt-2 text-sm text-gray-400 flex justify-between">
                                <span>FPS: {fps}</span>
                                <span>Processing: {processingTime.toFixed(1)}ms</span>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            <div className="mt-8 bg-gray-800 p-4 rounded-lg shadow-lg">
                <h2 className="text-xl font-semibold text-gray-200 mb-4">Controls</h2>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div>
                        <label className="block text-gray-200 mb-2">Kernel Size</label>
                        <select
                            value={kernelSize}
                            onChange={(e) => setKernelSize(Number(e.target.value) as KernelSize)}
                            className="w-full px-3 py-2 bg-gray-700 text-white rounded focus:outline-none focus:ring-2 focus:ring-accent"
                            disabled={isProcessing} // Disable while processing
                        >
                            <option value={3}>3×3</option>
                            <option value={5}>5×5</option>
                            <option value={7}>7×7</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-gray-200 mb-2">Output Format</label>
                        <select
                            value={outputFormat}
                            onChange={(e) => setOutputFormat(e.target.value as OutputFormat)}
                            className="w-full px-3 py-2 bg-gray-700 text-white rounded focus:outline-none focus:ring-2 focus:ring-accent"
                            disabled={isProcessing}
                        >
                            <option value="magnitude">Magnitude</option>
                            <option value="normalized">Normalized</option>
                            <option value="direction">Direction</option>
                            <option value="directionColor">Direction (Color)</option>
                            <option value="x">X-Gradient</option>
                            <option value="y">Y-Gradient</option>
                        </select>
                    </div>

                    <div className="flex items-center">
                        <input
                            type="checkbox"
                            id="useGrayscale"
                            checked={useGrayscale}
                            onChange={(e) => setUseGrayscale(e.target.checked)}
                            className="w-4 h-4 mr-2 accent-accent"
                            disabled={isProcessing}
                        />
                        <label htmlFor="useGrayscale" className="text-gray-200">
                            Convert to grayscale before processing
                        </label>
                    </div>

                    <div className="flex items-center">
                        <input
                            type="checkbox"
                            id="enhanceContrast"
                            checked={enhanceContrastEnabled}
                            onChange={(e) => setEnhanceContrastEnabled(e.target.checked)}
                            className="w-4 h-4 mr-2 accent-accent"
                        />
                        <label htmlFor="enhanceContrast" className="text-gray-200">
                            Enhance contrast of the result
                        </label>
                    </div>

                    <div className="flex items-center">
                        <input
                            type="checkbox"
                            id="lowQualityMode"
                            checked={lowQualityMode}
                            onChange={(e) => setLowQualityMode(e.target.checked)}
                            className="w-4 h-4 mr-2 accent-accent"
                            disabled={isProcessing}
                        />
                        <label htmlFor="lowQualityMode" className="text-gray-200">
                            Low quality mode (better performance)
                        </label>
                    </div>
                </div>

                <div className="mt-4 flex gap-4">
                    {isStreaming ? (
                        <button
                            onClick={stopStream}
                            className={`px-4 py-2 text-white rounded shadow transition-colors ${isProcessing ? 'bg-red-600 hover:bg-red-500' : 'bg-gray-500 cursor-not-allowed'}`}
                            disabled={!isProcessing} // Only allow stopping if processing
                        >
                            Stop Camera
                        </button>
                    ) : (
                        <button
                            onClick={startStream}
                            className="px-4 py-2 bg-accent text-white rounded shadow hover:bg-accent-light transition-colors"
                        >
                            Start Camera
                        </button>
                    )}
                </div>
            </div>
        </div>
    );
};

export default VideoProcessor; 
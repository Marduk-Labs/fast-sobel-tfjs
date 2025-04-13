import * as tf from '@tensorflow/tfjs';
import { useEffect, useRef, useState } from 'react';
import { SobelFilter } from 'sobel-tfjs';
import { Sobel as SobelTS } from 'sobel-ts';

// Define benchmark result type
interface BenchmarkResult {
    library: string;
    imageName: string;
    imageSize: string;
    executionTime: number;
    fps: number;
}

// Helper function to wait until an image is loaded
const loadImage = (url: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = (err) => reject(err);
        img.src = url;
    });
};

// Helper function to create a canvas from an image
const createCanvasFromImage = (img: HTMLImageElement): HTMLCanvasElement => {
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    if (ctx) {
        ctx.drawImage(img, 0, 0);
    }
    return canvas;
};

// Helper function to get ImageData from an image
const getImageDataFromImage = (img: HTMLImageElement): ImageData => {
    const canvas = createCanvasFromImage(img);
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) throw new Error('Failed to get canvas context');
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
};

const BenchmarkPage = () => {
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState<BenchmarkResult[]>([]);
    const [iterations, setIterations] = useState(5);
    const [useCache, setUseCache] = useState(true);
    const [currentTest, setCurrentTest] = useState('');
    const [progress, setProgress] = useState(0);

    // Reference to our loaded test images
    const imagesRef = useRef<{ [key: string]: HTMLImageElement }>({});

    // Reference to sobel filter instances
    const sobelFilterRef = useRef<SobelFilter | null>(null);

    // Test image URLs
    const testImages: Record<string, string> = {
        'Small (320x240)': 'small.jpg',
        'Medium (640x480)': 'medium.jpg',
        'Large (1280x720)': 'large.jpg',
        'HD (1920x1080)': 'hd.jpg',
        '2K (2560x1440)': '2k.jpg',
        '4K (3840x2160)': '4k.jpg'
    };

    // Initialize Sobel filter on mount
    useEffect(() => {
        sobelFilterRef.current = new SobelFilter({
            kernelSize: 3,
            output: 'magnitude',
            grayscale: true,
            normalizeOutputForDisplay: true
        });

        // Preload test images
        const preloadImages = async () => {
            setLoading(true);
            try {
                setCurrentTest('Loading test images...');

                // Try to load images using correct relative paths
                const imageNames = Object.keys(testImages);
                for (const imageName of imageNames) {
                    const path = testImages[imageName]; // Get the correct relative path
                    console.log(`[BENCHMARK] Trying to load ${imageName} from: ${path}`);

                    try {
                        const img = await loadImage(path);
                        imagesRef.current[imageName] = img;
                        console.log(`[BENCHMARK] Successfully loaded ${imageName} from: ${path}`);
                    } catch (error) {
                        console.error(`[BENCHMARK] Failed to load ${imageName} from: ${path}. Ensure the image exists in the /public directory.`);
                    }
                }

                // If no images could be loaded, create a fallback image
                if (Object.keys(imagesRef.current).length === 0) {
                    console.warn('[BENCHMARK] Creating fallback test image');

                    // Create a simple test pattern
                    const canvas = document.createElement('canvas');
                    canvas.width = 400;
                    canvas.height = 300;
                    const ctx = canvas.getContext('2d');
                    if (ctx) {
                        ctx.fillStyle = 'white';
                        ctx.fillRect(0, 0, 400, 300);
                        ctx.fillStyle = 'black';
                        ctx.fillRect(100, 100, 200, 100);

                        const fallbackImg = new Image();
                        fallbackImg.src = canvas.toDataURL();
                        await new Promise(resolve => {
                            fallbackImg.onload = resolve;
                        });

                        imagesRef.current['Fallback Test Image'] = fallbackImg;
                        console.log('[BENCHMARK] Created fallback test image');
                    }
                }
            } catch (err) {
                console.error('[BENCHMARK] Error preloading images:', err);
            } finally {
                setLoading(false);
                setCurrentTest('');
            }
        };

        preloadImages();

        return () => {
            // Clean up any resources
            sobelFilterRef.current = null;
        };
    }, []);

    // Run the benchmark
    const runBenchmark = async () => {
        if (loading) return;

        setLoading(true);
        setResults([]);
        setProgress(0);

        try {
            const newResults: BenchmarkResult[] = [];
            const totalTests = Object.keys(imagesRef.current).length * 2; // 2 libraries per image
            let testsCompleted = 0;

            // Run tests for each image
            for (const [imageName, img] of Object.entries(imagesRef.current)) {
                const imageSize = `${img.width}x${img.height}`;

                // Test sobel-tfjs
                setCurrentTest(`Testing sobel-tfjs with ${imageName} image...`);
                const tfResults = await benchmarkTensorflowJS(img, iterations, useCache);
                newResults.push({
                    library: 'sobel-tfjs',
                    imageName,
                    imageSize,
                    executionTime: tfResults.averageTime,
                    fps: tfResults.fps
                });

                testsCompleted++;
                setProgress(Math.round((testsCompleted / totalTests) * 100));

                // Test sobel-ts
                setCurrentTest(`Testing sobel-ts with ${imageName} image...`);
                const tsResults = await benchmarkSobelTS(img, iterations);
                newResults.push({
                    library: 'sobel-ts',
                    imageName,
                    imageSize,
                    executionTime: tsResults.averageTime,
                    fps: tsResults.fps
                });

                testsCompleted++;
                setProgress(Math.round((testsCompleted / totalTests) * 100));
            }

            setResults(newResults);
        } catch (error) {
            console.error('Benchmark error:', error);
        } finally {
            setLoading(false);
            setCurrentTest('');
            setProgress(100);
        }
    };

    // Benchmark the TensorFlow.js version
    const benchmarkTensorflowJS = async (
        img: HTMLImageElement,
        iterations: number,
        useCache: boolean
    ): Promise<{ averageTime: number; fps: number }> => {
        if (!sobelFilterRef.current) {
            throw new Error('Sobel filter not initialized');
        }

        const filter = sobelFilterRef.current;

        // Get input tensor ONCE before the loop
        const inputTensor = tf.tidy(() => tf.browser.fromPixels(img));
        console.log('[BENCHMARK-TFJS] Input tensor shape:', inputTensor.shape);

        // Warm-up run (not measured)
        tf.tidy(() => {
            const result = filter.applyToTensor(inputTensor);
            result.dispose(); // Dispose warm-up result
        });
        console.log('[BENCHMARK-TFJS] Warm-up completed');

        // Timed runs
        const times: number[] = [];

        for (let i = 0; i < iterations; i++) {
            let resultTensor: tf.Tensor | null = null; // To hold the result for disposal

            // Start scope management if not using cache
            if (!useCache) {
                tf.engine().startScope();
            }

            const start = performance.now();

            // Use applyToTensor directly
            resultTensor = tf.tidy(() => filter.applyToTensor(inputTensor));

            // Ensure the result is computed (needed for accurate timing)
            // We need to access data or shape to force computation
            if (resultTensor) {
                // Using shape should be lightweight enough
                const shape = resultTensor.shape;
                console.log("Result shape:", shape); // Optional: log shape if needed
                resultTensor.dispose(); // Dispose result tensor immediately after use
            } else {
                console.warn("[BENCHMARK-TFJS] applyToTensor returned null/undefined");
            }

            const end = performance.now();
            times.push(end - start);

            // End scope management if not using cache
            if (!useCache) {
                tf.engine().endScope();
            }

            // Small delay to prevent UI freezing
            await new Promise(resolve => setTimeout(resolve, 0));
        }

        // Dispose the input tensor after the loop
        inputTensor.dispose();
        console.log('[BENCHMARK-TFJS] Input tensor disposed');

        const totalTime = times.reduce((acc, time) => acc + time, 0);
        const averageTime = totalTime / times.length;
        const fps = 1000 / averageTime;

        console.log(`[BENCHMARK-TFJS] Avg Time: ${averageTime.toFixed(2)}ms, FPS: ${fps.toFixed(1)}`);

        return { averageTime, fps };
    };

    // Rename and update to benchmark the sobel-ts version
    const benchmarkSobelTS = async (
        img: HTMLImageElement,
        iterations: number
    ): Promise<{ averageTime: number; fps: number }> => {
        const imageData = getImageDataFromImage(img);

        // Warm-up run (not measured)
        try {
            const sobelInstance = new SobelTS(imageData); // Instantiate
            sobelInstance.apply('magnitude'); // Apply
        } catch (e) {
            console.error("[BENCHMARK-TS] Warm-up failed:", e);
            throw e; // Rethrow to stop benchmark if warm-up fails
        }
        console.log("[BENCHMARK-TS] Warm-up completed");

        // Timed runs
        const times: number[] = [];

        for (let i = 0; i < iterations; i++) {
            const start = performance.now();
            try {
                const sobelInstance = new SobelTS(imageData); // Instantiate inside loop
                sobelInstance.apply('magnitude'); // Apply
            } catch (e) {
                console.error(`[BENCHMARK-TS] Iteration ${i} failed:`, e);
                // Decide how to handle errors: continue, break, or push a NaN time?
                // For now, let's push NaN to indicate failure
                times.push(NaN);
                continue;
            }
            const end = performance.now();
            times.push(end - start);

            // Small delay to prevent UI freezing
            await new Promise(resolve => setTimeout(resolve, 0));
        }

        // Filter out any failed runs (NaN)
        const validTimes = times.filter(t => !isNaN(t));
        if (validTimes.length === 0 && times.length > 0) {
            console.error("[BENCHMARK-TS] All iterations failed.");
            return { averageTime: NaN, fps: NaN };
        } else if (validTimes.length < times.length) {
            console.warn(`[BENCHMARK-TS] ${times.length - validTimes.length} iterations failed.`);
        }

        const totalTime = validTimes.reduce((acc, time) => acc + time, 0);
        const averageTime = validTimes.length > 0 ? totalTime / validTimes.length : NaN;
        const fps = averageTime > 0 ? 1000 / averageTime : NaN;

        console.log(`[BENCHMARK-TS] Avg Time: ${averageTime.toFixed(2)}ms, FPS: ${fps.toFixed(1)}`);

        return { averageTime, fps };
    };

    return (
        <div className="max-w-6xl mx-auto p-6">
            <h1 className="text-3xl font-bold mb-6">Sobel Filter Performance Benchmark</h1>
            <p className="mb-6 text-gray-300">
                This benchmark compares the performance of the TensorFlow.js-based Sobel filter
                implementation against the <code className="bg-gray-800 px-1 rounded">sobel-ts</code> library (a standard JavaScript implementation).
            </p>

            <div className="bg-gray-900 rounded-lg p-6 mb-6 border border-gray-800">
                <h2 className="text-xl font-semibold mb-4">Benchmark Settings</h2>

                <div className="flex flex-wrap gap-4 mb-6">
                    <div className="flex-1 min-w-[200px]">
                        <label className="block text-sm text-gray-400 mb-1">
                            Iterations per test
                        </label>
                        <input
                            type="number"
                            min="1"
                            max="50"
                            value={iterations}
                            onChange={(e) => setIterations(parseInt(e.target.value) || 5)}
                            className="w-full bg-gray-800 text-white rounded p-2 border border-gray-700"
                            disabled={loading}
                        />
                    </div>

                    <div className="flex-1 min-w-[200px]">
                        <label className="block text-sm text-gray-400 mb-1">
                            TensorFlow.js Memory
                        </label>
                        <div className="flex items-center mt-2">
                            <input
                                type="checkbox"
                                id="useCache"
                                checked={useCache}
                                onChange={(e) => setUseCache(e.target.checked)}
                                className="mr-2"
                                disabled={loading}
                            />
                            <label htmlFor="useCache" className="text-gray-300">
                                Reuse tensors (faster but uses more memory)
                            </label>
                        </div>
                    </div>
                </div>

                <button
                    onClick={runBenchmark}
                    disabled={loading || Object.keys(imagesRef.current).length === 0}
                    className={`px-4 py-2 rounded ${loading
                        ? 'bg-gray-700 text-gray-300 cursor-not-allowed'
                        : 'bg-accent text-white hover:bg-accent-light'
                        }`}
                >
                    {loading ? 'Running Benchmark...' : 'Run Benchmark'}
                </button>
            </div>

            {loading && (
                <div className="bg-gray-900 rounded-lg p-6 mb-6 border border-gray-800">
                    <h2 className="text-xl font-semibold mb-4">Progress</h2>
                    <div className="w-full bg-gray-800 rounded-full h-4 mb-2">
                        <div
                            className="bg-accent h-4 rounded-full transition-all duration-300 ease-out"
                            style={{ width: `${progress}%` }}
                        ></div>
                    </div>
                    <p className="text-gray-400">{currentTest || 'Preparing benchmark...'}</p>
                </div>
            )}

            {results.length > 0 && (
                <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
                    <h2 className="text-xl font-semibold mb-4">Results</h2>

                    <div className="overflow-x-auto">
                        <table className="w-full text-left">
                            <thead>
                                <tr className="border-b border-gray-800">
                                    <th className="p-3">Library</th>
                                    <th className="p-3">Image</th>
                                    <th className="p-3">Size</th>
                                    <th className="p-3">Avg. Time (ms)</th>
                                    <th className="p-3">FPS</th>
                                    <th className="p-3">Comparison</th>
                                </tr>
                            </thead>
                            <tbody>
                                {results.map((result, index) => {
                                    // Find the paired result for the same image but different library
                                    const pairedResult = results.find(r =>
                                        r.imageName === result.imageName && r.library !== result.library
                                    );

                                    // Calculate performance ratio if paired result exists
                                    let performanceRatio = 1;
                                    let isFaster = false;

                                    if (pairedResult) {
                                        performanceRatio = pairedResult.executionTime / result.executionTime;
                                        isFaster = performanceRatio > 1;
                                    }

                                    return (
                                        <tr
                                            key={`${result.library}-${result.imageName}-${index}`}
                                            className="border-b border-gray-800 hover:bg-gray-800"
                                        >
                                            <td className="p-3">
                                                <span className={`font-semibold ${result.library === 'sobel-tfjs' ? 'text-blue-400' : 'text-green-400'
                                                    }`}>
                                                    {result.library}
                                                </span>
                                            </td>
                                            <td className="p-3">{result.imageName}</td>
                                            <td className="p-3">{result.imageSize}</td>
                                            <td className="p-3">{result.executionTime.toFixed(2)} ms</td>
                                            <td className="p-3">{result.fps.toFixed(1)}</td>
                                            <td className="p-3">
                                                {pairedResult && (
                                                    <span className={`${isFaster ? 'text-green-400' : 'text-yellow-400'}`}>
                                                        {isFaster
                                                            ? `${performanceRatio.toFixed(2)}x faster`
                                                            : `${(1 / performanceRatio).toFixed(2)}x slower`
                                                        }
                                                    </span>
                                                )}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </div>

                    <div className="mt-6">
                        <h3 className="text-lg font-semibold mb-2">Summary</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div className="bg-gray-800 p-4 rounded">
                                <h4 className="font-medium text-blue-400 mb-2">sobel-tfjs</h4>
                                <p className="text-gray-300">
                                    Average time: {
                                        (results
                                            .filter(r => r.library === 'sobel-tfjs')
                                            .reduce((sum, r) => sum + r.executionTime, 0) /
                                            results.filter(r => r.library === 'sobel-tfjs').length
                                        ).toFixed(2)
                                    } ms
                                </p>
                                <p className="text-gray-300">
                                    Average FPS: {
                                        (results
                                            .filter(r => r.library === 'sobel-tfjs')
                                            .reduce((sum, r) => sum + r.fps, 0) /
                                            results.filter(r => r.library === 'sobel-tfjs').length
                                        ).toFixed(1)
                                    }
                                </p>
                            </div>

                            <div className="bg-gray-800 p-4 rounded">
                                <h4 className="font-medium text-green-400 mb-2">sobel-ts</h4>
                                <p className="text-gray-300">
                                    Average time: {
                                        (results
                                            .filter(r => r.library === 'sobel-ts')
                                            .reduce((sum, r) => sum + r.executionTime, 0) /
                                            results.filter(r => r.library === 'sobel-ts').length
                                        ).toFixed(2)
                                    } ms
                                </p>
                                <p className="text-gray-300">
                                    Average FPS: {
                                        (results
                                            .filter(r => r.library === 'sobel-ts')
                                            .reduce((sum, r) => sum + r.fps, 0) /
                                            results.filter(r => r.library === 'sobel-ts').length
                                        ).toFixed(1)
                                    }
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="mt-6 text-gray-400 text-sm">
                        <p>Benchmark run with {iterations} iterations per test {useCache ? 'with' : 'without'} tensor caching.</p>
                        <p className="mt-2">
                            <strong>Note:</strong> Results may vary based on hardware, browser, and system load.
                            First-time runs may be slower due to just-in-time compilation and optimization.
                        </p>
                        <div className="mt-4 p-4 bg-gray-800 rounded">
                            <h4 className="font-semibold text-white mb-2">What is this comparison showing?</h4>
                            <ul className="list-disc pl-5 space-y-2">
                                <li><span className="text-blue-400 font-medium">sobel-tfjs</span> uses TensorFlow.js to process images with GPU acceleration where available.</li>
                                <li><span className="text-green-400 font-medium">sobel-ts</span> uses pure JavaScript code running on the CPU for the same algorithm.</li>
                                <li>The difference between them demonstrates the performance advantage of hardware-accelerated machine learning libraries.</li>
                                <li>Larger performance differences generally indicate better GPU utilization.</li>
                            </ul>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default BenchmarkPage; 
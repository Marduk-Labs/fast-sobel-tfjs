import * as tf from '@tensorflow/tfjs'
import { useEffect, useState } from 'react'
import ImageProcessor from './components/ImageProcessor'

function App() {
    const [tfReady, setTfReady] = useState(false)

    useEffect(() => {
        // Initialize TensorFlow.js
        const initTf = async () => {
            await tf.ready()
            console.log('TensorFlow.js initialized')
            setTfReady(true)
        }

        initTf()
    }, [])

    return (
        <main className="min-h-screen p-6 md:p-12">
            <header className="text-center max-w-4xl mx-auto mb-16">
                <h1 className="text-4xl md:text-6xl font-bold mb-4 bg-gradient-to-r from-white to-accent-light bg-clip-text text-transparent">
                    Sobel Edge Detection
                </h1>
                <p className="text-lg md:text-xl text-gray-300 mb-6">
                    Powered by TensorFlow.js
                </p>
            </header>

            {!tfReady ? (
                <div className="text-center py-10">
                    <div className="inline-block h-10 w-10 animate-spin rounded-full border-4 border-accent border-r-transparent"></div>
                    <p className="mt-4 text-lg text-gray-300">Initializing TensorFlow.js...</p>
                </div>
            ) : (
                <ImageProcessor />
            )}

            <footer className="mt-8 text-center text-sm text-gray-500">
                <p>
                    Built with ❤️ by Marduk Labs
                </p>
            </footer>
        </main>
    )
}

export default App 
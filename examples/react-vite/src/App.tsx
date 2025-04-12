import * as tf from '@tensorflow/tfjs'
import { useEffect, useState } from 'react'
import ImageProcessor from './components/ImageProcessor'
import VideoProcessor from './components/VideoProcessor'

function App() {
    const [tfReady, setTfReady] = useState(false)
    const [activeTab, setActiveTab] = useState<'image' | 'video'>('image')

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
            <header className="text-center max-w-4xl mx-auto mb-8">
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
                <>
                    <div className="max-w-6xl mx-auto mb-8">
                        <div className="flex border-b border-gray-700">
                            <button
                                className={`px-6 py-3 focus:outline-none ${activeTab === 'image' ? 'text-accent border-b-2 border-accent font-medium' : 'text-gray-400 hover:text-gray-200'}`}
                                onClick={() => setActiveTab('image')}
                            >
                                Image Processing
                            </button>
                            <button
                                className={`px-6 py-3 focus:outline-none ${activeTab === 'video' ? 'text-accent border-b-2 border-accent font-medium' : 'text-gray-400 hover:text-gray-200'}`}
                                onClick={() => setActiveTab('video')}
                            >
                                Video Processing
                            </button>
                        </div>
                    </div>

                    {activeTab === 'image' ? <ImageProcessor /> : <VideoProcessor />}
                </>
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
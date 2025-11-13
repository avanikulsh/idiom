import Link from "next/link";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-16">
        <header className="text-center mb-16">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Cross-Lingual Idiom Matcher
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Extract multi-word expressions from movie subtitles and find semantic similarities
            with English idioms using multilingual embeddings
          </p>
        </header>

        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
            <div className="text-3xl mb-4">📚</div>
            <h3 className="text-xl font-semibold mb-2">English Idioms Corpus</h3>
            <p className="text-gray-600 dark:text-gray-300">
              Start with a comprehensive collection of common English idioms
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
            <div className="text-3xl mb-4">🎬</div>
            <h3 className="text-xl font-semibold mb-2">MWE Extraction</h3>
            <p className="text-gray-600 dark:text-gray-300">
              Extract multi-word expressions from movie subtitles in Spanish, Hindi, and more
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
            <div className="text-3xl mb-4">🔍</div>
            <h3 className="text-xl font-semibold mb-2">Semantic Matching</h3>
            <p className="text-gray-600 dark:text-gray-300">
              Use multilingual transformers to find semantically similar expressions
            </p>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 p-8 rounded-lg shadow-lg mb-16">
          <h2 className="text-2xl font-bold mb-4">Project Structure</h2>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold mb-2 text-blue-600 dark:text-blue-400">Research Notebooks</h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li>• Data Exploration</li>
                <li>• MWE Extraction</li>
                <li>• Semantic Similarity Analysis</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold mb-2 text-purple-600 dark:text-purple-400">Python Pipeline</h3>
              <ul className="space-y-2 text-gray-600 dark:text-gray-300">
                <li>• Subtitle parsing utilities</li>
                <li>• spaCy-based MWE extractors</li>
                <li>• Sentence transformer similarity</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 p-6 rounded-lg">
          <h2 className="text-2xl font-bold mb-4">Getting Started</h2>
          <ol className="space-y-3 text-gray-700 dark:text-gray-300">
            <li>
              <strong>1. Set up Python environment:</strong>
              <code className="block mt-1 p-2 bg-gray-100 dark:bg-gray-800 rounded text-sm">
                python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
              </code>
            </li>
            <li>
              <strong>2. Download spaCy models:</strong>
              <code className="block mt-1 p-2 bg-gray-100 dark:bg-gray-800 rounded text-sm">
                python -m spacy download en_core_web_sm es_core_news_sm
              </code>
            </li>
            <li>
              <strong>3. Add your data:</strong> Place English idioms in <code>data/raw/english_idioms/</code>
              and subtitle files in <code>data/raw/subtitles/</code>
            </li>
            <li>
              <strong>4. Run notebooks:</strong> Open Jupyter and explore the notebooks in order
            </li>
          </ol>
        </div>

        <div className="text-center mt-12">
          <Link
            href="/results"
            className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold px-8 py-3 rounded-lg transition-colors"
          >
            View Results
          </Link>
        </div>
      </div>
    </div>
  );
}

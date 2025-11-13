"use client";

import { useState } from "react";
import Link from "next/link";

// This is a placeholder component. In production, you would:
// 1. Create an API route to read JSON files from data/results/
// 2. Fetch the data dynamically
// 3. Display the actual matching results

interface Match {
  english_idiom: string;
  foreign_mwe: string;
  language: string;
  similarity_score: number;
}

export default function ResultsPage() {
  const [selectedLanguage, setSelectedLanguage] = useState<string>("all");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [minScore, setMinScore] = useState<number>(0.6);

  // Placeholder data - will be replaced with actual API call
  const sampleMatches: Match[] = [
    {
      english_idiom: "break the ice",
      foreign_mwe: "romper el hielo",
      language: "spanish",
      similarity_score: 0.92,
    },
    {
      english_idiom: "piece of cake",
      foreign_mwe: "pan comido",
      language: "spanish",
      similarity_score: 0.87,
    },
  ];

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <Link
            href="/"
            className="text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
          >
            ← Back to Home
          </Link>
        </div>

        <header className="mb-8">
          <h1 className="text-4xl font-bold mb-2">Idiom Matching Results</h1>
          <p className="text-gray-600 dark:text-gray-300">
            Explore semantically similar expressions across languages
          </p>
        </header>

        {/* Filters */}
        <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4">Filters</h2>
          <div className="grid md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-2">Language</label>
              <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
              >
                <option value="all">All Languages</option>
                <option value="spanish">Spanish</option>
                <option value="hindi">Hindi</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Min Similarity Score: {minScore.toFixed(2)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={minScore}
                onChange={(e) => setMinScore(parseFloat(e.target.value))}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Search Idiom</label>
              <input
                type="text"
                placeholder="Search English idiom..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700"
              />
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 p-6 rounded-lg mb-8">
          <h3 className="font-semibold mb-2">How to use this page:</h3>
          <ol className="list-decimal list-inside space-y-1 text-sm text-gray-700 dark:text-gray-300">
            <li>Run the Jupyter notebooks to generate matching results</li>
            <li>Results will be saved as JSON in <code>data/results/</code></li>
            <li>Create an API route in <code>app/api/results/route.ts</code> to load the JSON</li>
            <li>Update this page to fetch and display the actual results</li>
          </ol>
        </div>

        {/* Results Table */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-100 dark:bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                    English Idiom
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                    Foreign MWE
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                    Language
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider">
                    Similarity Score
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {sampleMatches.length > 0 ? (
                  sampleMatches.map((match, index) => (
                    <tr
                      key={index}
                      className="hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                    >
                      <td className="px-6 py-4 whitespace-nowrap font-medium">
                        {match.english_idiom}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {match.foreign_mwe}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className="px-2 py-1 text-xs font-semibold rounded-full bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200">
                          {match.language}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2 mr-2">
                            <div
                              className="bg-green-500 h-2 rounded-full"
                              style={{ width: `${match.similarity_score * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-sm">
                            {match.similarity_score.toFixed(3)}
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td
                      colSpan={4}
                      className="px-6 py-8 text-center text-gray-500 dark:text-gray-400"
                    >
                      No results found. Run the notebooks to generate matching data.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Statistics */}
        <div className="grid md:grid-cols-3 gap-6 mt-8">
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              Total Matches
            </div>
            <div className="text-3xl font-bold">--</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              Avg Similarity
            </div>
            <div className="text-3xl font-bold">--</div>
          </div>
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">
              Languages
            </div>
            <div className="text-3xl font-bold">--</div>
          </div>
        </div>
      </div>
    </div>
  );
}

const CopyPlugin = require("copy-webpack-plugin");
const path = require('path');

module.exports = {
  entry: {
    main: './index.js',
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: (pathData) => {
      return pathData.chunk.name === 'main' ? 'bundle.js' : '[name]/bundle.js';
    },
    clean: true,
  },
  plugins: [new CopyPlugin({
    // Use copy plugin to copy *.wasm to output folder.
    patterns: [
      { from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' },
      { from: 'index.html', to: 'index.html' },
    ]
  })],
  mode: 'development'
};

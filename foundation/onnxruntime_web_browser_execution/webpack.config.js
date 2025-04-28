const CopyPlugin = require("copy-webpack-plugin");
const path = require('path');

module.exports = {
  entry: './index.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js',
  },
  plugins: [new CopyPlugin({
    // Use copy plugin to copy *.wasm to output folder.
    patterns: [
      { from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' },
      'index.html',
      'model/*.onnx'
    ]
  })],
  mode: 'development'
};

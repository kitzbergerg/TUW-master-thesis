const CopyPlugin = require("copy-webpack-plugin");
const path = require('path');

module.exports = {
  entry: {
    main: './index.js',
  },
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js'
  },
  plugins: [new CopyPlugin({
    patterns: [
      { from: 'index.html', to: 'index.html' },
    ]
  })],
  mode: 'development'
};

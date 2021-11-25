const webpack = require('webpack');
const config = {
<<<<<<< HEAD
    entry: __dirname + '/js/index.js',
=======
    entry:  __dirname + '/js/index.js',
>>>>>>> b4ef37e (add react (unfinished))
    output: {
        path: __dirname + '/dist',
        filename: 'bundle.js',
    },
    resolve: {
        extensions: ['.js', '.jsx', '.css']
    },
<<<<<<< HEAD

    module: {
        rules: [
            {
                test: /\.(js|jsx)?/,
                exclude: /node_modules/,
                use: 'babel-loader'
            },
            {
                test: /\.(png|svg|jpg|gif)$/,
                use: 'file-loader'
            },
            {
                test: /\.json$/,
                loader: 'json-loader'
            }
        ],
=======
  
    module: {
        rules: [
			{
			test: /\.(js|jsx)?/,
				exclude: /node_modules/,
				use: 'babel-loader'		
			},
			{
				test: /\.(png|svg|jpg|gif)$/,
				use: 'file-loader'
			}			
		]
>>>>>>> b4ef37e (add react (unfinished))
    }
};
module.exports = config;